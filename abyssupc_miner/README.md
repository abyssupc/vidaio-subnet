# abyssupc_miner

Custom optimization layer for SN85 Vidaio compression mining. **Not upstream code.**

## Modules

```
abyssupc_miner/
├── __init__.py
├── local_validator.py      # Validator-equivalent offline scoring
├── encoder.py              # Smart 2-pass adaptive encoder (libsvtav1 + VMAF binary search)
├── validator_data_logger.py # Captures validator requests for offline training
├── miner_wrapper.py        # Production miner entry point (replaces stock wrapper)
└── README.md
```

## Run

```bash
cd <repo_root>
python -m abyssupc_miner.miner_wrapper \
    --wallet.name <W> --wallet.hotkey <H> \
    --subtensor.network test --netuid 85 \
    --axon.port 8091 --logging.debug
```

Env vars:
- `VIDAIO_TASK`               — `COMPRESSION` (default) or `UPSCALING`
- `VIDAIO_MAX_LEN`            — `5` or `10` (content length cap)
- `VIDAIO_CAPTURE`            — `1` (default) enable validator data capture
- `VIDAIO_USE_SMART_ENCODER`  — `1` (default) use in-process smart encode
- `VIDAIO_LOG_DIR`            — path for synapses.jsonl + uid_metrics.jsonl
- `VIDAIO_CAPTURE_DIR`        — path for captured ref videos
- `VIDAIO_CAPTURE_MAX_GB`     — disk cap for captured data (default 20)
- `VIDAIO_BIN_DIR`            — path to static `ffmpeg-vmaf` + `ffprobe-vmaf`

## How it differs from stock

1. **In-process compression** — monkey-patches `services.miner_utilities.miner_utils.video_compressor` to call `encoder.smart_encode` directly. No HTTP roundtrip to `services/compress/server.py`.
2. **VMAF-target binary search** — finds highest cq that passes threshold + buffer (3 by default). Typically 2-4 attempts, 5-15s total.
3. **libsvtav1 default** — better quality/byte than av1_nvenc on most content (5-10 VMAF advantage).
4. **Validator data capture** — async downloads + deduplicates every reference video for offline training corpus.
5. **Per-synapse JSONL logging** — synapse type, validator UID, payload, timing, success.
6. **Periodic chain snapshot** — UID/incentive/emission/rank every 60s into `uid_metrics.jsonl`.

## Stock services still required

- `services/miner_utilities/file_deletion_server.py` — bucket TTL cleanup
- Redis (system service) — job state + queues

Stock `services/compress/server.py` is NOT required when `VIDAIO_USE_SMART_ENCODER=1`.

## Calibration

Defaults derived from offline testing on 11 varied Pexels samples (2026-05-12):

```python
# Per-threshold initial CQ (binary search refines per-clip)
DEFAULT_CQ = {
    "av1":  {85: 36, 89: 30, 93: 26},
    "hevc": {85: 26, 89: 22, 93: 20},
}
DEFAULT_BUFFER = 3.0  # VMAF safety buffer above threshold
```

Observed: avg S_f thr85=0.40, thr89=0.33, thr93=0.29 across varied content (vs stock ~0.11).

Recalibrate after collecting real validator data (`validator_data_logger` builds corpus).
