#!/bin/sh
# Keeps yt-dlp current (YouTube changes often) in the data volume, then starts the app.
set -e
mkdir -p /data/bin /data/downloads /data/outputs /data/models /data/data
if [ ! -x /data/bin/yt-dlp ]; then
    curl -fsSL https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /data/bin/yt-dlp && chmod +x /data/bin/yt-dlp || echo "could not download yt-dlp (will retry on next start)"
fi
/data/bin/yt-dlp -U >/dev/null 2>&1 || true
exec "$@"
