# Transcript

Podcast YouTube link -> SRT with speaker names (`Name: text` per cue) using OpenRouter. English only in the UI for now.

## Setup (Ubuntu / Debian)

```bash
sudo apt update && sudo apt install -y git ffmpeg nodejs python3 python3-venv curl
git clone git@github.com:Bimalbhattarai01/Transcript.git
cd Transcript
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdir -p bin && curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o bin/yt-dlp && chmod +x bin/yt-dlp
cp .env.example .env
nano .env
```

## Setup (macOS)

```bash
brew install git ffmpeg node python
git clone git@github.com:Bimalbhattarai01/Transcript.git
cd Transcript
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdir -p bin && curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o bin/yt-dlp && chmod +x bin/yt-dlp
cp .env.example .env
nano .env
```

In `.env` set `OPENROUTER_API_KEY=` to your key.

## Run the web UI

```bash
cd Transcript
. .venv/bin/activate
python3 ui.py
```

Open http://127.0.0.1:8765. Paste one link (or switch to **Many videos** for a list, or a playlist/channel link), pick a start/end time and a model, click **Generate SRT**.
Jobs go into a queue stored in SQLite (`data/app.db`); the **History** sidebar lists every job, and a job interrupted by a restart resumes. SRTs are also written to `outputs/`.

For speaker labels install the extra backend once: `pip install sherpa-onnx numpy` (fast, no account needed; its models download to `models/` on first use).
`DIARIZER=pyannote` uses the slower, more accurate pyannote instead (`pip install -r requirements-diarize.txt`, needs `HF_TOKEN` and the model terms accepted on huggingface.co).

## Run from the command line

```bash
. .venv/bin/activate

# test: 10 minutes starting at minute 5
python3 transcribe.py episode.mp3 --start 300 --limit-seconds 600

# full episode
python3 transcribe.py episode.mp3

# whole folder
python3 transcribe.py /path/to/episodes
```

## Run it on a server (VPS)

1. Copy the project to the server, e.g. `/opt/transcript` (`git clone` or `rsync`), then `sudo APP_DIR=/opt/transcript bash deploy/setup.sh`.
   It installs ffmpeg/node, a venv with `requirements-vps.txt` (no torch), yt-dlp, and a systemd service `transcribe-ui`.
2. Edit `/opt/transcript/.env`: `OPENROUTER_API_KEY`, `UI_USER` (default `admin`) and a long `UI_PASSWORD` (single login). The server refuses to listen on a non-loopback address without a password.
3. `sudo systemctl start transcribe-ui` and watch `journalctl -u transcribe-ui -f`.
4. **Simplest and touches nothing else:** open it through an SSH tunnel from your own computer: `ssh -L 8765:127.0.0.1:8765 user@your-server`, then browse http://127.0.0.1:8765. No proxy change, no public port, nothing exposed.
   A public URL (optional, later): the app listens on `127.0.0.1:8765` only. Put a reverse proxy in front: `deploy/nginx.conf` (add it to your existing nginx, then `certbot --nginx -d your.domain`) or `deploy/Caddyfile`.
   If the proxy runs in Docker it cannot reach the host's `127.0.0.1`: set `UI_HOST=172.17.0.1` (the Docker bridge address; `UI_PASSWORD` is then mandatory), point the proxy at `http://172.17.0.1:8765`, and make sure the firewall does not expose port 8765 to the internet.

The service runs at low priority (`Nice=10`, `CPUQuota=300%`, `MemoryMax=4G`, `DIARIZE_THREADS=3`) so other services on the same machine stay responsive; adjust in `deploy/transcribe-ui.service`.

**YouTube may block a datacenter IP** ("Sign in to confirm you're not a bot"). Export `cookies.txt` from a logged-in browser (e.g. the "Get cookies.txt LOCALLY" extension), upload it to the server and set `YT_COOKIES=/opt/transcript/cookies.txt`; or set `YT_PROXY`. Keep yt-dlp fresh: `bin/yt-dlp -U`.

Back up `data/app.db` (it holds the whole history and every SRT). `KEEP_AUDIO=0` (set in the service) deletes each mp3 after its job so hundreds of episodes don't fill the disk.

**Private repo on the server:** create a read-only deploy key there, add the `.pub` on GitHub (repo Settings -> Deploy keys), then clone with it:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/transcript_deploy -N "" && cat ~/.ssh/transcript_deploy.pub
GIT_SSH_COMMAND="ssh -i ~/.ssh/transcript_deploy -o IdentitiesOnly=yes" git clone git@github.com:Bimalbhattarai01/Transcript.git /opt/transcript
```
