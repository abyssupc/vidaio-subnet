"""
Production encoder for SN85 miner.

Plugs into miner_wrapper.py — replaces the default services/compress/server.py call.
Implements adaptive 2-pass with VMAF feedback:

  Pass 1: Encode at initial CQ (per-threshold default)
  Quick check: if VMAF buffer comfortable AND ratio > 5x → ship
  Otherwise:
    Pass 2: VMAF binary search to find optimal CQ
    Pass 3: Format/size sanity check before upload

Time budget: 135s validator timeout
  - 1-pass happy path: ~3-5s
  - 2-pass with binary search: ~15-30s

Output: presigned URL of compressed file uploaded to miner's bucket.
"""
from __future__ import annotations
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Relative import — local_validator lives in same abyssupc_miner package
from .local_validator import (
    FFMPEG, FFPROBE,
    measure_vmaf, validate_dist_encoding_settings,
    get_video_info, parse_fps,
    COMPRESSION_RATE_WEIGHT, COMPRESSION_VMAF_WEIGHT, SOFT_THRESHOLD_MARGIN,
)
# When the package is imported, REPO is already on sys.path via local_validator
from services.scoring.scoring_function import calculate_compression_score

from loguru import logger


# ── Defaults derived from offline testing (May 2026) ─────────────────────────

# Per-threshold initial CQ (median target from sweep results)
# These are starting points; binary search refines per-clip.
DEFAULT_CQ = {
    "av1": {
        85: 36,   # threshold 85: aggressive CQ for compression
        89: 30,   # threshold 89: middle ground
        93: 26,   # threshold 93: conservative quality
    },
    "hevc": {
        85: 26,
        89: 22,
        93: 20,
    },
}

# VMAF safety buffer above threshold (don't ride the line)
DEFAULT_BUFFER = 3.0


@dataclass
class EncodeResult:
    ok: bool
    output_path: Optional[str]
    final_cq: Optional[int]
    vmaf: Optional[float]
    size_bytes: Optional[int]
    ratio_x: Optional[float]
    estimated_s_f: Optional[float]
    attempts: int
    total_time_s: float
    reason: str
    history: list


# ── Encoder backends ──────────────────────────────────────────────────────────

def encode_av1(input_p: Path, output_p: Path, cq: int) -> dict:
    """libsvtav1 encode (best quality:speed for AV1)."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_p),
        "-c:v", "libsvtav1",
        "-preset", "6",
        "-crf", str(cq), "-b:v", "0",
        "-svtav1-params", "tune=0:lookahead=120:scd=1",
        "-pix_fmt", "yuv420p",
        "-vf", "setsar=1:1",
        "-c:a", "copy", "-movflags", "+faststart",
        str(output_p),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return {
        "ok": proc.returncode == 0 and output_p.exists() and output_p.stat().st_size > 0,
        "duration_s": round(time.time() - t0, 2),
        "error": proc.stderr[:300] if proc.returncode != 0 else None,
    }


def encode_hevc(input_p: Path, output_p: Path, cq: int) -> dict:
    """libx265 encode (best quality:speed for HEVC)."""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_p),
        "-c:v", "libx265",
        "-preset", "medium",
        "-crf", str(cq),
        "-pix_fmt", "yuv420p",
        "-vf", "setsar=1:1",
        "-c:a", "copy", "-movflags", "+faststart",
        str(output_p),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return {
        "ok": proc.returncode == 0 and output_p.exists() and output_p.stat().st_size > 0,
        "duration_s": round(time.time() - t0, 2),
        "error": proc.stderr[:300] if proc.returncode != 0 else None,
    }


ENCODERS = {"av1": encode_av1, "hevc": encode_hevc}


# ── Smart encoder with adaptive 2-pass ────────────────────────────────────────

def smart_encode(
    ref_path: Path,
    output_path: Path,
    vmaf_threshold: float,
    target_codec: str = "av1",
    codec_mode: str = "CRF",
    target_bitrate: float = 10.0,
    buffer: float = DEFAULT_BUFFER,
    max_attempts: int = 4,
    verbose: bool = True,
) -> EncodeResult:
    """
    Adaptive 2-pass encoder.

    Pass 1: encode at default CQ per (codec, threshold).
    If VMAF in [threshold+buffer, threshold+12] AND C < 0.5 → ship (1-pass happy).
    Otherwise: VMAF binary search around the initial cq.
    """
    t_start = time.time()
    codec = target_codec.lower()
    if codec not in ENCODERS:
        return EncodeResult(False, None, None, None, None, None, None, 0,
                            round(time.time() - t_start, 2),
                            f"unsupported codec: {codec}", [])

    encode_fn = ENCODERS[codec]
    target_vmaf = vmaf_threshold + buffer

    # Initial CQ from defaults
    cq_init = DEFAULT_CQ.get(codec, {}).get(int(vmaf_threshold), 32)
    if codec == "hevc":
        cq_lo, cq_hi = 18, 36
    else:  # av1
        cq_lo, cq_hi = 24, 46

    ref_size = ref_path.stat().st_size
    work_dir = output_path.parent / f".tmp_{output_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_cq, best_vmaf, best_size, best_C, best_tmp = None, 0, 0, 1.0, None
    attempts = 0
    cq = cq_init

    while attempts < max_attempts and cq_lo <= cq_hi:
        attempts += 1
        tmp = work_dir / f"a{attempts}_cq{cq}.mp4"
        if verbose:
            logger.info(f"  pass {attempts}: cq={cq} (range [{cq_lo}, {cq_hi}])")

        enc = encode_fn(ref_path, tmp, cq)
        if not enc["ok"]:
            logger.warning(f"  encode fail at cq={cq}: {enc.get('error')[:120]}")
            cq_hi = cq - 1
            cq = (cq_lo + cq_hi) // 2
            continue

        # Measure VMAF
        vmaf_r = measure_vmaf(str(tmp), str(ref_path))
        if not vmaf_r["ok"]:
            logger.warning(f"  VMAF fail at cq={cq}")
            cq_hi = cq - 1
            cq = (cq_lo + cq_hi) // 2
            continue

        vmaf = vmaf_r["harmonic_mean"]
        size = tmp.stat().st_size
        C = size / ref_size

        history.append({
            "attempt": attempts, "cq": cq, "vmaf": round(vmaf, 2),
            "size_kb": round(size / 1024, 1), "C": round(C, 4),
            "encode_s": enc["duration_s"], "vmaf_s": vmaf_r["duration_s"],
        })

        if verbose:
            logger.info(f"    → VMAF={vmaf:.2f}  size={size/1024:.1f}KB  C={C:.4f}")

        # CATASTROPHIC GUARD: output >= input → never ship
        if C >= 1.0:
            cq_hi = cq - 1
            cq = (cq_lo + cq_hi) // 2
            continue

        # HARD FAIL GUARD: C >= 0.80 → reject (would get S_f=0)
        if C >= 0.80:
            cq_hi = cq - 1
            cq = (cq_lo + cq_hi) // 2
            continue

        # Threshold check
        if vmaf >= target_vmaf:
            # Passing — track as best and try to compress more
            if cq > (best_cq or 0):
                best_cq, best_vmaf, best_size, best_C, best_tmp = cq, vmaf, size, C, tmp
            cq_lo = cq + 1

            # 1-pass happy path: comfortable margin AND good compression
            if vmaf <= target_vmaf + 8 and C < 0.50:
                if verbose:
                    logger.info(f"  ✓ sweet zone — stopping")
                break
        else:
            # Below target → need lower cq (higher quality)
            cq_hi = cq - 1

        cq = (cq_lo + cq_hi) // 2

    total_time = round(time.time() - t_start, 2)

    if best_tmp is None or not best_tmp.exists():
        # Cleanup
        for f in work_dir.glob("*.mp4"):
            try: f.unlink()
            except: pass
        try: work_dir.rmdir()
        except: pass
        return EncodeResult(False, None, None, None, None, None, None, attempts,
                            total_time, "no passing cq found", history)

    # Promote best to final output
    best_tmp.rename(output_path)
    # Cleanup remaining tmp files
    for f in work_dir.glob("*.mp4"):
        try: f.unlink()
        except: pass
    try: work_dir.rmdir()
    except: pass

    # Validate against validator's gate one final time
    ok, why = validate_dist_encoding_settings(
        str(output_path), str(ref_path),
        task="compression", target_codec=codec,
        codec_mode=codec_mode, target_bitrate=target_bitrate,
    )
    if not ok:
        return EncodeResult(False, str(output_path), best_cq, best_vmaf, best_size,
                            ref_size / best_size, None, attempts, total_time,
                            f"GATE FAIL: {why}", history)

    # Estimate final S_f
    s_f, _, _, _ = calculate_compression_score(
        vmaf_score=best_vmaf,
        compression_rate=best_C,
        vmaf_threshold=vmaf_threshold,
        compression_weight=COMPRESSION_RATE_WEIGHT,
        quality_weight=COMPRESSION_VMAF_WEIGHT,
        soft_threshold_margin=SOFT_THRESHOLD_MARGIN,
    )

    return EncodeResult(
        ok=True,
        output_path=str(output_path),
        final_cq=best_cq,
        vmaf=round(best_vmaf, 2),
        size_bytes=best_size,
        ratio_x=round(ref_size / best_size, 2),
        estimated_s_f=round(s_f, 4),
        attempts=attempts,
        total_time_s=total_time,
        reason=why,
        history=history,
    )


# ── CLI for testing ──────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--threshold", type=float, default=89)
    ap.add_argument("--codec", default="av1")
    ap.add_argument("--mode", default="CRF")
    ap.add_argument("--bitrate", type=float, default=10.0)
    ap.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    ap.add_argument("--max-attempts", type=int, default=4)
    args = ap.parse_args()

    ref = Path(args.ref).resolve()
    out_dir = Path(os.environ.get(
        "VIDAIO_ENCODER_OUT_DIR",
        str(Path.home() / "bittensor-projects" / "sn85" / "test" / "outputs" / "encoder_test")
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"smart_{ref.stem}_{args.codec}_thr{int(args.threshold)}.mp4"

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Production smart encoder")
    print(f"  ref={ref.name}  codec={args.codec}  threshold={args.threshold}  buffer=+{args.buffer}")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    result = smart_encode(
        ref_path=ref, output_path=output_path,
        vmaf_threshold=args.threshold,
        target_codec=args.codec, codec_mode=args.mode, target_bitrate=args.bitrate,
        buffer=args.buffer, max_attempts=args.max_attempts,
    )

    print(f"\n━━━ Result ━━━")
    if not result.ok:
        print(f"  ❌ FAIL: {result.reason}")
        print(f"  attempts: {result.attempts}, time: {result.total_time_s}s")
    else:
        print(f"  ✓ output: {result.output_path}")
        print(f"  cq: {result.final_cq}  VMAF: {result.vmaf}  ratio: {result.ratio_x}x")
        print(f"  size: {result.size_bytes/1024:.1f} KB")
        print(f"  estimated S_f: {result.estimated_s_f}")
        print(f"  attempts: {result.attempts}, time: {result.total_time_s}s")
        if result.estimated_s_f > 0.74:
            print(f"  🌟 BONUS ZONE")
        elif result.estimated_s_f >= 0.4:
            print(f"  ✓ PASS")
        elif result.estimated_s_f > 0.08:
            print(f"  ⚠ weak")
        else:
            print(f"  ✗ FAIL tier")


if __name__ == "__main__":
    main()
