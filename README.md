# Transcript

Nepali podcast -> SRT (+ English SRT) using OpenRouter.

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

Open http://127.0.0.1:8765, paste a YouTube link, click **Generate SRT**. Files are saved in `outputs/`.

## Run from the command line

```bash
. .venv/bin/activate

# test: 10 minutes starting at minute 5
python3 transcribe.py episode.mp3 --start 300 --limit-seconds 600

# full episode (Nepali .srt + English .en.srt)
python3 transcribe.py episode.mp3

# whole folder
python3 transcribe.py /path/to/episodes

# only Nepali, no cleanup / translation
python3 transcribe.py episode.mp3 --no-clean --no-translate

# translate / clean an SRT you already have
python3 translate.py outputs/episode.srt
python3 translate.py outputs/episode.srt --clean
```
