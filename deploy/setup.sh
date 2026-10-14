#!/usr/bin/env bash
# One-time server setup (Debian/Ubuntu). Put the code in $APP_DIR first (git clone or rsync), then run as root:
#   sudo APP_DIR=/opt/transcript bash deploy/setup.sh
set -euo pipefail
APP_DIR="${APP_DIR:-/opt/transcript}"; APP_USER="${APP_USER:-srt}"
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
[ -f "$APP_DIR/ui.py" ] || { echo "ui.py not found in $APP_DIR - copy the project there first"; exit 1; }

apt-get update
apt-get install -y ffmpeg python3-venv python3-pip curl ca-certificates nodejs
node -v | grep -qE '^v(2[0-9]|[3-9][0-9])' || echo "WARNING: yt-dlp wants Node 20+ (found $(node -v)); install a newer Node, e.g. from nodesource.com"

id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
cd "$APP_DIR"
mkdir -p bin data downloads outputs models
[ -f .env ] || { cp .env.example .env; chmod 600 .env; }
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-vps.txt      # sherpa-onnx speaker backend; no torch, no HuggingFace token needed
curl -fsSL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o bin/yt-dlp && chmod +x bin/yt-dlp
[ -f static/Inter.woff2 ] || echo "note: static/Inter.woff2 missing (the page falls back to the system font)"
chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

sed "s#/opt/transcript#$APP_DIR#g; s#^User=srt#User=$APP_USER#" deploy/transcribe-ui.service > /etc/systemd/system/transcribe-ui.service
systemctl daemon-reload
systemctl enable transcribe-ui
cat <<MSG

Done. Next:
  1. Edit $APP_DIR/.env : OPENROUTER_API_KEY, UI_PASSWORD (long!), optionally YT_COOKIES
  2. systemctl start transcribe-ui && journalctl -u transcribe-ui -f
  3. Put HTTPS in front: deploy/Caddyfile (Caddy) or an nginx server block -> proxy_pass http://127.0.0.1:8765
MSG
