"""
Validator data logger — capture full request payload + reference video
that validator sends to miner. Build offline dataset for optimization.

Designed to be INJECTED into miner_wrapper.py via monkey-patch.
DOES NOT modify production code.

Data captured per synapse:
  - All payload fields (URL, threshold, codec, mode, bitrate, round_id, validator hotkey/UID)
  - Downloaded reference video (presigned URL → local file before revocation)
  - Metadata JSON sibling
  - Hash for dedup (validator may resend same clip)

Storage:
  ~/bittensor-projects/sn85/research/captured/
  ├── index.jsonl                  # one line per captured request
  ├── refs/<sha256_prefix>.mp4     # deduplicated reference videos
  └── metadata/<request_id>.json   # per-request metadata

Caps:
  - Max total disk: configurable (default 20 GB)
  - LRU-style pruning when over cap (oldest refs go first)
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger


# ── Config ───────────────────────────────────────────────────────────────────

CAPTURE_DIR = Path(os.environ.get(
    "VIDAIO_CAPTURE_DIR",
    str(Path.home() / "bittensor-projects" / "sn85" / "research" / "captured")
))
INDEX_FILE = CAPTURE_DIR / "index.jsonl"
REFS_DIR = CAPTURE_DIR / "refs"
META_DIR = CAPTURE_DIR / "metadata"

MAX_TOTAL_GB = float(os.environ.get("VIDAIO_CAPTURE_MAX_GB", "20"))
DOWNLOAD_TIMEOUT_S = 60

REFS_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


def _payload_dict(synapse) -> dict:
    """Extract all payload fields from a synapse (compression/upscaling)."""
    out: dict = {"synapse_type": type(synapse).__name__}
    try:
        if hasattr(synapse, "round_id"):
            out["round_id"] = synapse.round_id
        if hasattr(synapse, "job_id"):
            out["job_id"] = synapse.job_id
        if hasattr(synapse, "version") and synapse.version:
            v = synapse.version
            out["version"] = f"{v.major}.{v.minor}.{v.patch}"
        if hasattr(synapse, "miner_payload"):
            p = synapse.miner_payload
            for field in ["reference_video_url", "vmaf_threshold", "target_codec",
                          "codec_mode", "target_bitrate", "task_type",
                          "maximum_optimized_size_mb"]:
                if hasattr(p, field):
                    out[field] = getattr(p, field)
    except Exception as e:
        out["_payload_parse_error"] = str(e)
    return out


async def _download_ref(url: str, dest_path: Path,
                        timeout_s: int = DOWNLOAD_TIMEOUT_S) -> dict:
    """Async download of reference video. Returns metadata."""
    t0 = time.time()
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s, sock_connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    return {"ok": False, "status": resp.status,
                            "error": f"HTTP {resp.status}"}
                content_len = int(resp.headers.get("Content-Length", 0))
                with open(dest_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(2 * 1024 * 1024):
                        f.write(chunk)
        size = dest_path.stat().st_size
        return {
            "ok": True,
            "size_bytes": size,
            "content_length_header": content_len,
            "download_time_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "download_time_s": round(time.time() - t0, 2)}


def _hash_url(url: str) -> str:
    """Stable hash of URL (used as filename pre-content-hash for dedup)."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _file_sha256(path: Path) -> str:
    """SHA256 of file contents for content-hash dedup."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prune_old_refs():
    """If captured/refs/ exceeds MAX_TOTAL_GB, delete oldest files."""
    max_bytes = MAX_TOTAL_GB * 1024 * 1024 * 1024
    files = sorted(REFS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    if total <= max_bytes:
        return
    while files and total > max_bytes * 0.9:  # prune down to 90% of cap
        oldest = files.pop(0)
        size = oldest.stat().st_size
        try:
            oldest.unlink()
            total -= size
        except Exception as e:
            logger.warning(f"prune fail: {e}")


async def capture_request(synapse, validator_uid: Optional[int] = None,
                          validator_hk: Optional[str] = None) -> dict:
    """
    Main entry point — call this from wrapped forward_*_requests handlers.
    Non-blocking: downloads + saves async, returns immediately.

    Returns: capture record (also appended to index.jsonl).
    """
    record = {
        "ts": time.time(),
        "validator_uid": validator_uid,
        "validator_hk": (validator_hk or "")[:14],
        **_payload_dict(synapse),
    }

    url = record.get("reference_video_url")
    if not url:
        # Some synapse types don't carry a ref URL (LengthCheck, TaskWarrant)
        # Still log the request itself
        record["captured"] = False
        record["reason"] = "no reference_video_url"
        _append_index(record)
        return record

    # Determine target filename via URL hash
    url_hash = _hash_url(url)
    tmp_path = REFS_DIR / f".tmp_{url_hash}.mp4"

    # Download
    dl = await _download_ref(url, tmp_path)
    record["download"] = dl

    if not dl["ok"]:
        record["captured"] = False
        if tmp_path.exists():
            tmp_path.unlink()
        _append_index(record)
        return record

    # Content-hash dedup
    content_hash = _file_sha256(tmp_path)
    final_path = REFS_DIR / f"{content_hash[:16]}.mp4"
    if final_path.exists():
        # Already have this exact content — keep older copy
        tmp_path.unlink()
        record["captured"] = True
        record["dedup"] = True
        record["ref_file"] = final_path.name
        record["content_hash"] = content_hash
    else:
        tmp_path.rename(final_path)
        record["captured"] = True
        record["dedup"] = False
        record["ref_file"] = final_path.name
        record["content_hash"] = content_hash

    # Per-request metadata sidecar
    request_id = record.get("round_id") or record.get("job_id") or f"{int(record['ts']*1000)}"
    meta_path = META_DIR / f"{request_id}.json"
    with open(meta_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    # Append to index
    _append_index(record)

    # Best-effort prune
    try:
        _prune_old_refs()
    except Exception as e:
        logger.warning(f"prune err: {e}")

    return record


def _append_index(record: dict) -> None:
    """Append one line to index.jsonl. Best-effort, never raises."""
    try:
        with open(INDEX_FILE, "a") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"index append fail: {e}")


# ── Integration helper (monkey-patch onto Miner class) ───────────────────────

def wrap_forward_with_capture(orig_fn):
    """
    Decorator that captures synapse before forwarding to original handler.
    Wraps async miner forward functions like forward_compression_requests.
    """
    async def wrapped(self, synapse, *args, **kwargs):
        # Capture request (async background, doesn't block forward)
        try:
            val_hk = synapse.dendrite.hotkey if synapse.dendrite else None
            val_uid = None
            if val_hk and val_hk in self.metagraph.hotkeys:
                val_uid = self.metagraph.hotkeys.index(val_hk)
            # Fire-and-forget capture (don't await)
            asyncio.create_task(
                capture_request(synapse, validator_uid=val_uid, validator_hk=val_hk)
            )
        except Exception as e:
            logger.warning(f"capture launch fail: {e}")

        # Call original handler
        return await orig_fn(self, synapse, *args, **kwargs)

    wrapped.__name__ = getattr(orig_fn, "__name__", "wrapped_with_capture")
    return wrapped


# ── CLI for inspection of captured data ──────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Validator data logger / inspector")
    ap.add_argument("--show", action="store_true", help="Show captured index stats")
    ap.add_argument("--list", action="store_true", help="List captured files")
    args = ap.parse_args()

    print(f"Capture dir: {CAPTURE_DIR}")
    print(f"Refs:        {REFS_DIR}")
    print(f"Metadata:    {META_DIR}")
    print(f"Index:       {INDEX_FILE}")
    print()

    if args.show or args.list:
        if not INDEX_FILE.exists():
            print("(no index yet — has miner been running with capture enabled?)")
            return

        total_records = 0
        captured = 0
        by_type = {}
        by_validator = {}
        unique_content = set()
        with open(INDEX_FILE) as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                total_records += 1
                if rec.get("captured"):
                    captured += 1
                    if "content_hash" in rec:
                        unique_content.add(rec["content_hash"])
                st = rec.get("synapse_type", "?")
                by_type[st] = by_type.get(st, 0) + 1
                vh = rec.get("validator_hk", "?")[:8]
                by_validator[vh] = by_validator.get(vh, 0) + 1

        print(f"Total requests logged:    {total_records}")
        print(f"  with reference saved:   {captured}")
        print(f"  unique content hashes:  {len(unique_content)}")
        print()
        print(f"By synapse type:")
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t:40} {c}")
        print()
        print(f"By validator:")
        for v, c in sorted(by_validator.items(), key=lambda x: -x[1]):
            print(f"  {v:14} {c}")
        print()
        # Total disk used
        total_size = sum(f.stat().st_size for f in REFS_DIR.glob("*.mp4"))
        print(f"Refs storage: {total_size/1024/1024:.1f} MB ({len(list(REFS_DIR.glob('*.mp4')))} files)")


if __name__ == "__main__":
    main()
