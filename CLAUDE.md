# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Podcast YouTube link -> SRT with speaker names via OpenRouter (UI is English-only for now; the CLI still supports --lang ne). Flat Python scripts, only dependency is `requests`. No tests or linter configured. System deps: `ffmpeg`, `node` (yt-dlp needs a JS runtime for YouTube), `yt-dlp` in `bin/` (or on PATH).

## Commands

```bash
. .venv/bin/activate
cp .env.example .env            # set OPENROUTER_API_KEY (real env vars override .env)

python3 ui.py                   # web UI at http://127.0.0.1:8765, outputs in outputs/
python3 transcribe.py episode.mp3 --start 300 --limit-seconds 600   # quick 10-min test (writes *.test.srt)
python3 transcribe.py episode.mp3                                   # full: Nepali .srt + English .en.srt
python3 transcribe.py DIR                                           # batch a folder
python3 bakeoff.py EPISODE.mp3 CAPCUT.srt [--models a,b,c]          # compare STT models vs a CapCut reference SRT (WER/CER) -> out/
```

## Architecture

- `transcribe.py` — core pipeline (`process()`). ffmpeg silencedetect -> `speech_spans` -> `make_chunks` (cut at real silences, <= `--max-chunk` s, because OpenRouter times out ~60s/call and long audio causes Whisper-style drift) -> chunks transcribed in parallel via `/audio/transcriptions` (raw JSON cached in `<out>/.cache/<episode>[-tag]/`, so runs are resumable) -> `timed_words` maps chunk-local times back to wall-clock using the *speech-only* timeline (interpolates word times when the API returns none) -> `build_cues` packs words into cues -> `write_srt`. Cue limits are constants near the top (`LINE_CHARS`, `MAX_DUR`, ...). Existing `.srt` are skipped unless `--force`.
- `translate.py` — only `chat`/`parse_array`/`CLEAN_MODEL` are used now (by `transcribe.name_speakers`); the translate/clean pipeline is no longer wired into the CLI or UI.
- `ui.py` — stdlib `http.server` app with the HTML page embedded as the `PAGE` string. Jobs are rows in SQLite (`db.py`, `data/app.db`: queue + history + SRT text); one background `worker()` takes the oldest `queued` job and runs `run_one()` (yt-dlp download -> `transcribe.process()` with a hand-built `argparse.Namespace`, so new CLI args in `transcribe.py` must also be added there). Progress/log reach the DB through `JobIO` and the `transcribe.hooks` dict; `state["cost"]` gives per-job cost. API: `/api/jobs`, `/api/job/<id>`, `/api/result|file/<id>`, `/api/zip`, `/api/expand` (playlist/channel -> videos), `/api/enqueue|retry|cancel|delete`. Optional single-password login (`UI_PASSWORD`, signed cookie); it refuses a non-loopback `UI_HOST` without one. Deploy files are in `deploy/`.
- `db.py` — SQLite layer (WAL, one locked connection); on startup `running` jobs go back to `queued`.
- `diarize.py` — speaker turns for `--speakers`. `DIARIZER=fast` (sherpa-onnx, default when installed, models auto-download to `models/`) or `pyannote` (slower on CPU, needs `HF_TOKEN`); `transcribe.process()` runs it in a background thread alongside chunk transcription, then `smooth_speakers` and `name_speakers` (LLM names from the transcript) turn ids into `Name:` prefixes.
- `bakeoff.py` — standalone evaluation; also provides SRT parsing and WER/CER helpers (`norm`, `wer`, `cer`) that `transcribe.py --compare` imports.
- `env.py` — tiny `.env` loader, called at import by each script.

Gitignored: `.env`, `.cache/`, `out/`, `outputs/`, `downloads/`, `bin/`, `*.srt`, `failures.txt` (failed episodes are appended there).
