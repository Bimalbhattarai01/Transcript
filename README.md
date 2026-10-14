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
Use **Projects** (left sidebar, like ChatGPT) to keep each channel or show separate: pick a project when you queue a video, or let a playlist/channel link suggest one. Hover a transcript or project to delete it; open a transcript to move it to another project. Your name and **Log out** are in the menu at the top right.
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

## Run it on a server without Docker (systemd)

1. Copy the project to the server, e.g. `/opt/transcript` (`git clone` or `rsync`), then `sudo APP_DIR=/opt/transcript bash deploy/setup.sh`.
   It installs ffmpeg/node, a venv with `requirements-vps.txt` (no torch), yt-dlp, and a systemd service `transcribe-ui`.
2. Edit `/opt/transcript/.env`: `OPENROUTER_API_KEY`. Then create the team logins (see below); the service refuses to start until at least one user exists.
3. `sudo -u srt /opt/transcript/.venv/bin/python /opt/transcript/manage.py adduser alice`, then `sudo systemctl start transcribe-ui` and watch `journalctl -u transcribe-ui -f`.
4. **Simplest and touches nothing else:** open it through an SSH tunnel from your own computer: `ssh -L 8765:127.0.0.1:8765 user@your-server`, then browse http://127.0.0.1:8765. No proxy change, no public port, nothing exposed.
   A public URL (optional, later): the app listens on `127.0.0.1:8765` only. Put a reverse proxy in front: `deploy/nginx.conf` (add it to your existing nginx, then `certbot --nginx -d your.domain`) or `deploy/Caddyfile`.
   If the proxy runs in Docker it cannot reach the host's `127.0.0.1`: set `UI_HOST=172.17.0.1` (the Docker bridge address; login is then mandatory), point the proxy at `http://172.17.0.1:8765`, and make sure the firewall does not expose port 8765 to the internet.

The service runs at low priority (`Nice=10`, `CPUQuota=300%`, `MemoryMax=4G`, `DIARIZE_THREADS=3`) so other services on the same machine stay responsive; adjust in `deploy/transcribe-ui.service`.

**YouTube may block a datacenter IP** ("Sign in to confirm you're not a bot"). Export `cookies.txt` from a logged-in browser (e.g. the "Get cookies.txt LOCALLY" extension), upload it to the server and set `YT_COOKIES=/opt/transcript/cookies.txt`; or set `YT_PROXY`. Keep yt-dlp fresh: `bin/yt-dlp -U`.

Back up `data/app.db` (it holds the whole history and every SRT). `KEEP_AUDIO=0` (set in the service) deletes each mp3 after its job so hundreds of episodes don't fill the disk.

**Private repo on the server:** create a read-only deploy key there, add the `.pub` on GitHub (repo Settings -> Deploy keys), then clone with it:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/transcript_deploy -N "" && cat ~/.ssh/transcript_deploy.pub
GIT_SSH_COMMAND="ssh -i ~/.ssh/transcript_deploy -o IdentitiesOnly=yes" git clone git@github.com:Bimalbhattarai01/Transcript.git /opt/transcript
```

### Team logins

Accounts live in the database; there is no signup page, so only someone with shell access to the server can add people. History is shared by the whole team and each job shows who added it.

```bash
M="sudo -u srt /opt/transcript/.venv/bin/python /opt/transcript/manage.py"
$M adduser alice      # asks for a password (min 10 characters)
$M listusers
$M passwd alice       # also signs alice out everywhere
$M disable alice      # / enable alice  (takes effect immediately)
$M deluser alice
```

Update the server later with `cd /opt/transcript && git pull` (use the deploy key as above), `sudo chown -R srt:srt /opt/transcript`, `sudo systemctl restart transcribe-ui`. An existing database upgrades itself.

## Run it with Docker (recommended on the server)

**Quick deploy:** clone the repo on the server, then `bash deploy/vps-docker.sh` (checks Docker, port and disk, asks for the OpenRouter key, creates the first login, builds, starts, prints the SSH-tunnel command). The manual steps below do the same.

Self-contained: its own image and volume, nothing else on the host is touched. CPU is capped at 3 cores and 4 GB RAM in `docker-compose.yml`; change it there.

```bash
cp .env.example .env && nano .env        # OPENROUTER_API_KEY (the file is NOT copied into the image)
docker compose up -d --build
docker compose exec srt python manage.py adduser you@example.com    # asks for a password (min 10 chars); repeat per teammate
docker compose logs -f srt
```

- The container refuses to start until a user exists, so it can never come up without a login. Create the user right after the first `up` (it restarts by itself until then), or run `docker compose run --rm srt python manage.py adduser NAME` first.
- Port: published on `127.0.0.1:8765` only. nginx on the host proxies to it (`deploy/nginx.conf`). If your nginx runs in Docker, put `BIND_ADDR=172.17.0.1` in `.env` and proxy to `http://172.17.0.1:8765`; never publish it on `0.0.0.0` without the proxy and HTTPS.
- Data (database, downloads, outputs, speaker models, yt-dlp) lives in the `srt-data` volume; yt-dlp updates itself on every start. Back up with `docker run --rm -v srt-data:/d -v "$PWD":/b alpine tar czf /b/srt-data.tgz -C /d .`
- YouTube blocking the server ("Sign in to confirm you're not a bot" is common on datacenter IPs, and cookies are the fix): uncomment the `cookies.txt` volume line in `docker-compose.yml` and set `YT_COOKIES=/data/cookies.txt` in `.env`.
- Update: `git pull && docker compose up -d --build`. Team commands: `docker compose exec srt python manage.py listusers|passwd|disable|enable|deluser NAME`.
