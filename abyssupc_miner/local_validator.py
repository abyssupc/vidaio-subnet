"""
Local validator-equivalent scoring harness for offline miner optimization.

Imports the actual validator code from cloned repo where possible. Replaces only:
  - VMAF measurement: static `ffmpeg-vmaf` (libvmaf) instead of Docker libvmaf_cuda
    (same algorithm, same harmonic mean — only difference is GPU vs CPU)

Use this to:
  1. Test miner encoder output → predict the EXACT score validator would assign
  2. Sweep encoder configs offline, find optimal per (threshold, codec, content)
  3. Validate format compliance (encoding gate) before going live
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ─── Import actual validator code (zero replication) ────────────────────────
from services.scoring.scoring_function import calculate_compression_score
# We replicate validate_dist_encoding_settings + normalize_codec_family because
# the originals are inside a 3000-line FastAPI server with global state.
# But the LOGIC is identical — code copied verbatim with only the ffprobe path
# pointing to our static ffprobe (full libvmaf-enabled toolchain).

# Static ffmpeg/ffprobe with libvmaf compiled in
# (binaries live OUTSIDE the repo at ~/bittensor-projects/sn85/bin/)
_BIN_DIR = Path(os.environ.get(
    "VIDAIO_BIN_DIR", str(Path.home() / "bittensor-projects" / "sn85" / "bin")
))
FFMPEG = str(_BIN_DIR / "ffmpeg-vmaf")
FFPROBE = str(_BIN_DIR / "ffprobe-vmaf")

# Validator constants (services/scoring/server.py:29-33)
COMPRESSION_RATE_WEIGHT = 0.70  # NOT 0.8 from docs — code is truth
COMPRESSION_VMAF_WEIGHT = 0.30
SOFT_THRESHOLD_MARGIN = 5.0
FRAME_TOLERANCE = 5


# ─── Replicated helpers (from services/scoring/server.py) ────────────────────

def normalize_codec_family(codec_name: str) -> str:
    """Verbatim from server.py:1202."""
    c = codec_name.lower().strip()
    if any(v in c for v in ["av1", "libaom", "libsvtav1", "svt-av1"]):
        return "av1"
    if any(v in c for v in ["hevc", "h265", "x265", "libx265"]):
        return "hevc"
    if any(v in c for v in ["h264", "avc", "x264", "libx264"]):
        return "h264"
    if "vp9" in c or "libvpx-vp9" in c:
        return "vp9"
    if "vp8" in c or "libvpx" in c:
        return "vp8"
    return c


def get_video_info(path: str) -> dict:
    """One-shot ffprobe matching the validator's gate-check query."""
    cmd = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,profile,level,sample_aspect_ratio,pix_fmt,"
        "width,height,r_frame_rate,avg_frame_rate,color_space,color_primaries,color_transfer,bit_rate,nb_read_frames",
        "-show_entries", "format=tags=encoder,format_name,bit_rate",
        "-count_frames",
        "-of", "json",
        path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if p.returncode != 0:
        return {"error": p.stderr[:300]}
    return json.loads(p.stdout)


def parse_fps(fps_str: str) -> float:
    try:
        num, den = map(float, fps_str.split("/"))
        return num / den if den != 0 else 0.0
    except Exception:
        return 0.0


def validate_dist_encoding_settings(
    dist_path: str, ref_path: str, task: str = "compression",
    target_codec: str = "av1", codec_mode: str = "CRF", target_bitrate: float = 10.0,
) -> Tuple[bool, str]:
    """
    Validator's encoding gate. Returns (pass, reason).
    Verbatim logic from services/scoring/server.py:1238.
    """
    try:
        info = get_video_info(dist_path)
        if "error" in info:
            return False, f"ffprobe failed: {info['error']}"

        streams = info.get("streams", [])
        if not streams:
            return False, "no video stream"
        st = streams[0]
        fmt = info.get("format", {})

        codec = st.get("codec_name", "")
        profile = st.get("profile", "")
        sar = st.get("sample_aspect_ratio", "1:1")
        pix_fmt = st.get("pix_fmt", "")
        width = st.get("width", 0)
        height = st.get("height", 0)
        container = fmt.get("format_name", "")

        avg_fps = parse_fps(st.get("avg_frame_rate", "0/1"))
        bit_rate = st.get("bit_rate") or fmt.get("bit_rate")
        bit_rate_mbps = int(bit_rate) / 1_000_000 if bit_rate else None

        errors = []

        # Bitrate check (only for CBR/VBR; CRF skipped per server.py:1337)
        if codec_mode and codec_mode.upper() in ["CBR", "VBR"]:
            if bit_rate_mbps is None:
                errors.append("Bitrate missing for CBR/VBR mode")
            else:
                upper = target_bitrate * 1.10
                if bit_rate_mbps > upper:
                    errors.append(f"Bitrate {bit_rate_mbps:.2f} > target+10% ({upper:.2f}) Mbps")

        # FPS check (±0.3)
        ref_info = get_video_info(ref_path)
        ref_st = ref_info.get("streams", [{}])[0]
        ref_fps = parse_fps(ref_st.get("avg_frame_rate", "30/1"))
        if abs(avg_fps - ref_fps) > 0.3:
            errors.append(f"FPS mismatch: ref={ref_fps:.2f}, dist={avg_fps:.2f}")

        # Resolution (compression task requires exact match)
        if task == "compression":
            ref_w, ref_h = ref_st.get("width", 0), ref_st.get("height", 0)
            if (width, height) != (ref_w, ref_h):
                errors.append(f"Resolution mismatch: ref={ref_w}x{ref_h}, dist={width}x{height}")

        # Codec family
        det_family = normalize_codec_family(codec)
        exp_family = normalize_codec_family(target_codec) if task == "compression" else "hevc"
        if det_family != exp_family:
            errors.append(f"Codec family {det_family} != expected {exp_family}")

        # Container
        if "ivf" in container.lower():
            errors.append("Container must be MP4, got IVF")
        if container not in ["mov,mp4,m4a,3gp,3g2,mj2", "mp4", "isom"]:
            errors.append(f"Container must be MP4, got {container}")

        # Profile per codec
        if det_family == "av1" and profile != "Main":
            errors.append(f"AV1 profile must be Main, got {profile}")
        elif det_family == "hevc" and profile not in ["Main", "Main 10"]:
            errors.append(f"HEVC profile must be Main/Main 10, got {profile}")

        # SAR + pix_fmt mandatory
        if sar != "1:1":
            errors.append(f"SAR must be 1:1, got {sar}")
        if pix_fmt != "yuv420p":
            errors.append(f"pix_fmt must be yuv420p, got {pix_fmt}")

        if errors:
            return False, "; ".join(errors)

        br_str = f"{bit_rate_mbps:.2f}" if bit_rate_mbps else "?"
        return True, (
            f"valid ({codec} {profile}, {width}x{height}@{avg_fps}, "
            f"bitrate {br_str} Mbps, mode={codec_mode})"
        )

    except Exception as e:
        return False, f"validation error: {e}"


def measure_vmaf(dist_path: str, ref_path: str, subsample: int = 1) -> dict:
    """
    Measure VMAF using libvmaf via static ffmpeg.
    Matches validator's harmonic_mean pooling. Equivalent to libvmaf_cuda except CPU.
    """
    log_path = Path(dist_path).with_suffix(".vmaf.json")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", dist_path,
        "-i", ref_path,
        "-lavfi",
        (
            f"[0:v]setpts=PTS-STARTPTS,format=yuv420p[d];"
            f"[1:v]setpts=PTS-STARTPTS,format=yuv420p[r];"
            f"[d][r]libvmaf=n_subsample={subsample}:log_path={log_path}:log_fmt=json:pool=harmonic_mean"
        ),
        "-f", "null", "-",
    ]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    dt = time.time() - t0
    if p.returncode != 0:
        return {"ok": False, "error": p.stderr[:300], "duration_s": dt}
    try:
        data = json.load(open(log_path))
        v = data["pooled_metrics"]["vmaf"]
        return {
            "ok": True,
            "duration_s": round(dt, 2),
            "harmonic_mean": round(v["harmonic_mean"], 3),
            "mean": round(v["mean"], 3),
            "min": round(v["min"], 3),
            "max": round(v["max"], 3),
            "n_frames": len(data.get("frames", [])),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "duration_s": dt}


def count_frames(path: str) -> int:
    info = get_video_info(path)
    if "error" in info:
        return 0
    return int(info.get("streams", [{}])[0].get("nb_read_frames", 0))


# ─── The full validator scoring pipeline (single-call) ───────────────────────

def score_locally(
    dist_path: str,
    ref_path: str,
    vmaf_threshold: float,
    target_codec: str = "av1",
    codec_mode: str = "CRF",
    target_bitrate: float = 10.0,
) -> dict:
    """
    Run validator-equivalent scoring on (dist, ref) pair.
    Returns dict with score, reasoning, all intermediate metrics.
    """
    out: dict = {
        "dist_path": dist_path,
        "ref_path": ref_path,
        "vmaf_threshold": vmaf_threshold,
        "target_codec": target_codec,
        "codec_mode": codec_mode,
        "target_bitrate": target_bitrate,
        "pass_gate": False,
        "s_f": 0.0,
        "reason": "",
    }

    # 1. File existence + size
    if not os.path.exists(dist_path) or not os.path.exists(ref_path):
        out["reason"] = "missing file"
        return out
    dist_size = os.path.getsize(dist_path)
    ref_size = os.path.getsize(ref_path)
    if dist_size == 0:
        out["reason"] = "empty output"
        return out

    out["dist_size"] = dist_size
    out["ref_size"] = ref_size
    out["C"] = round(dist_size / ref_size, 4)
    out["ratio_x"] = round(ref_size / dist_size, 2) if dist_size else 0

    # 2. Validator's encoding gate (codec/profile/SAR/pix_fmt/etc.)
    ok, why = validate_dist_encoding_settings(
        dist_path, ref_path, task="compression",
        target_codec=target_codec, codec_mode=codec_mode, target_bitrate=target_bitrate,
    )
    out["gate_ok"] = ok
    out["gate_reason"] = why
    if not ok:
        out["reason"] = f"GATE FAIL: {why}"
        return out

    # 3. Frame count tolerance (±5)
    dist_frames = count_frames(dist_path)
    ref_frames = count_frames(ref_path)
    out["dist_frames"] = dist_frames
    out["ref_frames"] = ref_frames
    if abs(dist_frames - ref_frames) > FRAME_TOLERANCE:
        out["reason"] = f"frame count off by {abs(dist_frames - ref_frames)} > {FRAME_TOLERANCE}"
        return out

    # 4. Catastrophic protection (compression_rate >= 1.0 → -10 in miner_manager,
    #    we report S_f=0 here but flag it)
    if out["C"] >= 1.0:
        out["reason"] = f"OUTPUT >= INPUT (C={out['C']}) — CATASTROPHIC -10 PENALTY"
        out["catastrophic"] = True
        return out

    # 5. Hard fail: C >= 0.80 (less than 1.25x compression)
    if out["C"] >= 0.80:
        out["reason"] = f"C={out['C']} >= 0.80 (<1.25x compression)"
        return out

    # 6. VMAF measurement
    vmaf_result = measure_vmaf(dist_path, ref_path)
    out["vmaf"] = vmaf_result
    if not vmaf_result["ok"]:
        out["reason"] = f"VMAF measurement failed: {vmaf_result.get('error', '?')}"
        return out
    vmaf_score = vmaf_result["harmonic_mean"]
    out["vmaf_hmean"] = vmaf_score

    # 7. Final scoring (calling validator's actual scoring function)
    s_f, comp_c, qual_c, reason = calculate_compression_score(
        vmaf_score=vmaf_score,
        compression_rate=out["C"],
        vmaf_threshold=vmaf_threshold,
        compression_weight=COMPRESSION_RATE_WEIGHT,
        quality_weight=COMPRESSION_VMAF_WEIGHT,
        soft_threshold_margin=SOFT_THRESHOLD_MARGIN,
    )
    out["pass_gate"] = True
    out["s_f"] = round(s_f, 4)
    out["compression_component"] = round(comp_c, 4)
    out["quality_component"] = round(qual_c, 4)
    out["reason"] = reason

    # 8. Categorize tier
    if s_f > 0.74:
        out["tier"] = "★ BONUS"
    elif s_f >= 0.4:
        out["tier"] = "PASS"
    elif s_f > 0.08:
        out["tier"] = "weak"
    else:
        out["tier"] = "FAIL"

    return out


# ─── CLI for quick scoring ───────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Local validator-equivalent scoring")
    ap.add_argument("--dist", required=True, help="Distorted (miner output) video")
    ap.add_argument("--ref", required=True, help="Reference (validator's clip) video")
    ap.add_argument("--threshold", type=float, default=89.0, help="VMAF threshold")
    ap.add_argument("--codec", default="av1", help="Target codec family")
    ap.add_argument("--mode", default="CRF", help="Codec mode CRF/VBR/CBR")
    ap.add_argument("--bitrate", type=float, default=10.0, help="Target bitrate (Mbps)")
    args = ap.parse_args()

    result = score_locally(
        dist_path=args.dist,
        ref_path=args.ref,
        vmaf_threshold=args.threshold,
        target_codec=args.codec,
        codec_mode=args.mode,
        target_bitrate=args.bitrate,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
