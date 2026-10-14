#!/usr/bin/env bash
# One-command local setup (Linux / macOS). Safe to re-run.
#   bash setup.sh && . .venv/bin/activate && python3 ui.py
set -euo pipefail
cd "$(dirname "$0")"

missing=0
for c in python3 ffmpeg node; do
  command -v "$c" >/dev/null || { echo "missing: $c"; missing=1; }
done
if [ "$missing" = 1 ]; then
  echo "install them first:"
  echo "  Ubuntu/Debian: sudo apt install -y ffmpeg nodejs python3 python3-venv curl"
  echo "  macOS:         brew install ffmpeg node python"
  exit 1
fi

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements-vps.txt   # requests + sherpa-onnx (speaker labels)

if [ ! -x bin/yt-dlp ]; then
  mkdir -p bin
  curl -fsSL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o bin/yt-dlp
  chmod +x bin/yt-dlp
fi

[ -f .env ] || cp .env.example .env
grep -q '^OPENROUTER_API_KEY=your_' .env && echo "-> put your OpenRouter key in .env (OPENROUTER_API_KEY=...)"

echo "done. start the UI:  . .venv/bin/activate && python3 ui.py   (http://127.0.0.1:8765, no login locally)"
