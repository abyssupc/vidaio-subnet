"""
abyssupc miner_wrapper — PRODUCTION wrapper for SN85 Vidaio.

Replaces stock services/compress HTTP service with in-process smart_encode.
Adds:
  1. Env-driven task type / content length
  2. Per-synapse JSONL logging
  3. Periodic chain metric snapshots
  4. Validator data capture (downloads + dedup ref videos for offline training)
  5. Smart 2-pass adaptive encoder (libsvtav1 + VMAF binary search)

Run as module from repo root:
    cd ~/bittensor-projects/sn85/repo
    python -m abyssupc_miner.miner_wrapper --wallet.name X --wallet.hotkey Y --netuid 85 ...

Or via PM2 ecosystem (see prod/ecosystem.config.cjs).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

# Add repo root to PYTHONPATH so neurons/, services/, vidaio_subnet_core/ import
REPO = Path(__file__).parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOG_DIR = Path(os.environ.get(
    "VIDAIO_LOG_DIR",
    str(Path.home() / "bittensor-projects" / "sn85" / "prod" / "logs"),
)).expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)

SYN_LOG = LOG_DIR / "synapses.jsonl"
ERR_LOG = LOG_DIR / "errors.jsonl"

from loguru import logger
logger.add(LOG_DIR / "miner.log", rotation="50 MB", retention=10, level="DEBUG")


def jsonl_append(path: Path, record: dict) -> None:
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except Exception:
        pass


# Import upstream + our additions
import neurons.miner as miner_mod
from vidaio_subnet_core.protocol import (
    TaskType, ContentLength,
    VideoUpscalingProtocol, VideoCompressionProtocol,
    LengthCheckProtocol, TaskWarrantProtocol,
    VideoCompressionJobProtocol, VideoCompressionPollProtocol,
    VideoUpscalingJobProtocol, VideoUpscalingPollProtocol,
)

# Our optimization layer
from . import encoder as our_encoder
from . import validator_data_logger as vdl

# Vidaio's upstream miner_utils (we'll monkey-patch its video_compressor)
import services.miner_utilities.miner_utils as miner_utils
from services.miner_utilities.redis_utils import schedule_file_deletion
from vidaio_subnet_core.utilities import storage_client, download_video


# ── Env config (no source patching) ──────────────────────────────────────────
TASK_ENV = os.environ.get("VIDAIO_TASK", "COMPRESSION").upper()
MAX_LEN_ENV = os.environ.get("VIDAIO_MAX_LEN", "10")
CAPTURE_ON = os.environ.get("VIDAIO_CAPTURE", "1") == "1"
USE_SMART_ENCODER = os.environ.get("VIDAIO_USE_SMART_ENCODER", "1") == "1"

if TASK_ENV == "COMPRESSION":
    miner_mod.warrant_task = TaskType.COMPRESSION
elif TASK_ENV == "UPSCALING":
    miner_mod.warrant_task = TaskType.UPSCALING

if MAX_LEN_ENV == "10":
    miner_mod.MAX_CONTENT_LEN = ContentLength.TEN
elif MAX_LEN_ENV == "5":
    miner_mod.MAX_CONTENT_LEN = ContentLength.FIVE

logger.info(f"⚙ config: task={TASK_ENV} max_len={MAX_LEN_ENV}s "
            f"capture={CAPTURE_ON} smart_encoder={USE_SMART_ENCODER}")


# ── Smart encoder monkey-patch on video_compressor ───────────────────────────

WORK_DIR = LOG_DIR / "compress_work"
WORK_DIR.mkdir(exist_ok=True)


async def smart_video_compressor(payload_url: str, vmaf_threshold: float,
                                  target_codec: str = "av1",
                                  codec_mode: str = "CRF",
                                  target_bitrate: float = 10.0) -> str | None:
    """
    Replacement for upstream miner_utils.video_compressor that uses our
    in-process smart_encode (no HTTP roundtrip to services/compress).
    """
    t_start = time.time()
    job_id = uuid.uuid4().hex[:12]
    ref_path = None
    output_path = None
    try:
        logger.info(f"[{job_id}] smart_compress: thr={vmaf_threshold} codec={target_codec} mode={codec_mode}")

        # 1. Download reference video
        ref_path_str = await download_video(payload_url)
        ref_path = Path(ref_path_str)
        if not ref_path.exists():
            logger.warning(f"[{job_id}] download_video returned non-existent path")
            return None

        # 2. Smart encode
        output_path = WORK_DIR / f"out_{job_id}.mp4"
        result = our_encoder.smart_encode(
            ref_path=ref_path,
            output_path=output_path,
            vmaf_threshold=float(vmaf_threshold),
            target_codec=target_codec,
            codec_mode=codec_mode,
            target_bitrate=target_bitrate,
            buffer=our_encoder.DEFAULT_BUFFER,
            max_attempts=4,
            verbose=False,
        )

        if not result.ok:
            logger.warning(f"[{job_id}] smart_encode fail: {result.reason}")
            return None

        logger.info(f"[{job_id}] smart_encode ✓ cq={result.final_cq} VMAF={result.vmaf} "
                    f"ratio={result.ratio_x}x S_f≈{result.estimated_s_f} "
                    f"({result.attempts} attempts, {result.total_time_s}s)")

        # 3. Upload to bucket
        object_name = output_path.name
        await storage_client.upload_file(object_name, str(output_path))

        # 4. Get presigned URL
        sharing_link = await storage_client.get_presigned_url(object_name)
        if not sharing_link:
            logger.error(f"[{job_id}] presigned URL fetch failed")
            return None

        # 5. Schedule cleanup
        schedule_file_deletion(object_name)

        total = round(time.time() - t_start, 2)
        logger.info(f"[{job_id}] DONE in {total}s — URL: {sharing_link[:80]}")
        return sharing_link

    except Exception as e:
        logger.error(f"[{job_id}] smart_video_compressor error: {e}")
        traceback.print_exc()
        return None
    finally:
        # Local file cleanup
        for p in (ref_path, output_path):
            if p and p.exists():
                try: p.unlink()
                except: pass


# Install monkey-patch if enabled
if USE_SMART_ENCODER:
    miner_utils.video_compressor = smart_video_compressor
    logger.info("⚙ video_compressor monkey-patched → smart_video_compressor (in-process)")
else:
    logger.info("⚙ stock services/compress HTTP service kept (USE_SMART_ENCODER=0)")


# ── Per-synapse logging wrapper ──────────────────────────────────────────────

def _payload_dict(synapse: Any) -> dict:
    out: dict[str, Any] = {"synapse_type": type(synapse).__name__}
    try:
        if hasattr(synapse, "round_id"):
            out["round_id"] = synapse.round_id
        if hasattr(synapse, "job_id"):
            out["job_id"] = synapse.job_id
        if hasattr(synapse, "miner_payload"):
            p = synapse.miner_payload
            for field in ["reference_video_url", "vmaf_threshold", "target_codec",
                          "codec_mode", "target_bitrate", "task_type",
                          "maximum_optimized_size_mb"]:
                if hasattr(p, field):
                    out[field] = getattr(p, field)
            # truncate URL
            if "reference_video_url" in out and out["reference_video_url"]:
                out["ref_url"] = out["reference_video_url"][:150]
                del out["reference_video_url"]
        if hasattr(synapse, "max_content_length"):
            out["max_content_length"] = int(synapse.max_content_length)
    except Exception as e:
        out["payload_parse_error"] = str(e)
    return out


def wrap_forward(orig_fn):
    """Per-synapse JSONL logging + timing."""
    async def wrapped(self, synapse, *args, **kwargs):
        t0 = time.time()
        val_uid, val_hk = None, None
        try:
            val_hk = synapse.dendrite.hotkey if synapse.dendrite else None
            if val_hk and val_hk in self.metagraph.hotkeys:
                val_uid = self.metagraph.hotkeys.index(val_hk)
        except Exception:
            pass

        rec = {"ts": t0, "val_uid": val_uid, "val_hk": (val_hk or "")[:12],
               **_payload_dict(synapse)}

        try:
            result = await orig_fn(self, synapse, *args, **kwargs)
            rec["duration_s"] = round(time.time() - t0, 3)
            success = False
            if hasattr(result, "miner_response") and result.miner_response:
                url = getattr(result.miner_response, "optimized_video_url", "")
                success = bool(url)
                rec["output_url_set"] = success
                rec["output_url_prefix"] = (url or "")[:80]
            if hasattr(result, "job_response") and result.job_response:
                rec["job_accepted"] = bool(getattr(result.job_response, "accepted", False))
                success = success or rec["job_accepted"]
            if hasattr(result, "poll_response") and result.poll_response:
                rec["poll_status"] = getattr(result.poll_response, "status", "?")
                success = rec["poll_status"] in ("completed", "processing")
            if hasattr(result, "warrant_task") and result.warrant_task is not None:
                rec["warrant"] = str(result.warrant_task)
            if hasattr(result, "max_content_length"):
                rec["max_len_reply"] = int(result.max_content_length)
            rec["success"] = success
            jsonl_append(SYN_LOG, rec)
            return result
        except Exception as e:
            rec["duration_s"] = round(time.time() - t0, 3)
            rec["success"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["tb"] = traceback.format_exc()[:1500]
            jsonl_append(SYN_LOG, rec)
            jsonl_append(ERR_LOG, rec)
            raise

    wrapped.__name__ = getattr(orig_fn, "__name__", "wrapped_forward")
    return wrapped


# ── Wire validator data capture (outer layer) ────────────────────────────────

if CAPTURE_ON:
    wrap_capture = vdl.wrap_forward_with_capture
    logger.info("⚙ validator-data capture ENABLED")
else:
    wrap_capture = lambda fn: fn
    logger.info("⚙ validator-data capture disabled")


# Patch Miner class methods BEFORE instantiation
# Stack: capture (outer) → logging (inner) → original forward
Miner = miner_mod.Miner
for name in [
    "forward_upscaling_requests",
    "forward_compression_requests",
    "forward_length_check_requests",
    "forward_task_warrant_requests",
    "forward_compression_job_requests",
    "forward_compression_poll_requests",
    "forward_upscaling_job_requests",
    "forward_upscaling_poll_requests",
]:
    if hasattr(Miner, name):
        logged = wrap_forward(getattr(Miner, name))
        # Only wrap capture on real-work handlers (compression/upscaling requests + jobs)
        if any(tag in name for tag in ("compression_requests", "upscaling_requests",
                                        "compression_job", "upscaling_job")):
            wrapped = wrap_capture(logged)
        else:
            wrapped = logged
        setattr(Miner, name, wrapped)
        logger.info(f"⚙ wrapped {name}")


# ── Main loop with periodic chain metric snapshot ────────────────────────────

def main() -> None:
    logger.info("🚀 abyssupc-miner-wrapper starting")
    with Miner() as miner:
        logger.info(f"🆔 Miner UID = {miner.uid} on netuid {miner.config.netuid} "
                    f"(network={miner.config.subtensor.network})")

        last_snap = 0.0
        snap_path = LOG_DIR / "uid_metrics.jsonl"
        try:
            while True:
                now = time.time()
                if now - last_snap > 60:
                    try:
                        mg = miner.metagraph
                        u = miner.uid
                        rec = {
                            "ts": now,
                            "block": int(getattr(mg, "block", 0)),
                            "uid": u,
                            "incentive": float(mg.incentive[u]) if u < len(mg.incentive) else 0,
                            "emission": float(mg.emission[u]) if u < len(mg.emission) else 0,
                            "stake": float(mg.stake[u]) if u < len(mg.stake) else 0,
                            "consensus": float(mg.consensus[u]) if u < len(mg.consensus) else 0,
                            "trust": float(mg.validator_trust[u]) if u < len(mg.validator_trust) else 0,
                            "dividends": float(mg.dividends[u]) if u < len(mg.dividends) else 0,
                            "n_total": int(getattr(mg, "n", 0)),
                        }
                        jsonl_append(snap_path, rec)
                    except Exception as e:
                        jsonl_append(ERR_LOG, {"ts": now, "kind": "metric_snap_error", "err": str(e)})
                    last_snap = now
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("🛑 shutdown requested")


if __name__ == "__main__":
    main()
