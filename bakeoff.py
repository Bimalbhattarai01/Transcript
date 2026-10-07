#!/usr/bin/env python3
"""Compare OpenRouter STT models on one Nepali clip against a CapCut SRT.

usage: OPENROUTER_API_KEY=... python3 bakeoff.py EPISODE.mp3 CAPCUT.srt [--start 0] [--dur 600] [--models a,b,c]
Outputs in ./out/: ref.txt, <model>.json, <model>.srt, results.md
"""
import argparse, base64, json, os, re, subprocess, sys, time, unicodedata
from pathlib import Path
import requests
from env import load_env

load_env()
API = "https://openrouter.ai/api/v1/audio/transcriptions"
DEFAULT_MODELS = [
    "google/gemini-3.5-transcribe", "openai/gpt-transcribe", "google/chirp-3",
    "microsoft/mai-transcribe-2", "assemblyai/universal-3-5-pro", "deepgram/nova-3",
    "qwen/qwen3-asr-1.7b", "mistralai/voxtral-mini-transcribe", "openai/whisper-large-v3-turbo",
]
OUT = Path("out")


def ts_to_s(t):
    h, m, rest = t.split(":"); s, ms = rest.replace(",", ".").split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def s_to_ts(x):
    x = max(x, 0); ms = round(x * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def parse_srt(path):
    cues = []
    for blk in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8-sig").strip()):
        lines = blk.strip().splitlines()
        i = next((k for k, l in enumerate(lines) if "-->" in l), None)
        if i is None: continue
        a, b = [x.strip() for x in lines[i].split("-->")]
        cues.append((ts_to_s(a), ts_to_s(b), " ".join(lines[i + 1:])))
    return cues


def norm(text):
    text = unicodedata.normalize("NFC", text)
    # \w drops Devanagari combining marks (Mn/Mc), so filter by category: drop punctuation/symbols (incl. danda)
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(len(r), 1)


def cer(ref, hyp):
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    return edit_distance(r, h) / max(len(r), 1)


def repeat_flag(text):
    w = text.split()
    return any(w[i:i + 4] == w[i + 4:i + 8] == w[i + 8:i + 12] for i in range(max(len(w) - 12, 0)))


def cut_clip(src, start, dur):
    OUT.mkdir(exist_ok=True)
    clip = OUT / "clip.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(dur), "-i", src,
                    "-ac", "1", "-ar", "16000", "-b:a", "48k", str(clip)], check=True)
    return clip


def transcribe(model, clip, key):
    body = {"model": model, "language": "ne", "response_format": "verbose_json",
            "input_audio": {"data": base64.b64encode(clip.read_bytes()).decode(), "format": "mp3"}}
    t0 = time.time()
    r = requests.post(API, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=900)
    dt = time.time() - t0
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:300]}")
    return r.json(), dt


def words_or_segments(js):
    """Return list of (start, end, text); falls back to one cue if no timing info."""
    if js.get("segments"):
        return [(s["start"], s["end"], s["text"].strip()) for s in js["segments"]]
    if js.get("words"):
        out, cur = [], []
        for w in js["words"]:
            cur.append(w)
            tx = w.get("word") or w.get("text", "")
            if len(cur) >= 8 or tx.endswith(("।", ".", "?")):
                out.append((cur[0]["start"], cur[-1]["end"], " ".join((x.get("word") or x.get("text", "")).strip() for x in cur)))
                cur = []
        if cur:
            out.append((cur[0]["start"], cur[-1]["end"], " ".join((x.get("word") or x.get("text", "")).strip() for x in cur)))
        return out
    return [(0, 0, js.get("text", ""))]


def write_srt(cues, path):
    Path(path).write_text("".join(f"{i}\n{s_to_ts(a)} --> {s_to_ts(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(cues, 1)), encoding="utf-8")


def timing_offset(ref_cues, hyp_cues):
    """Mean abs start-time diff, matching each ref cue to the hyp cue nearest in time."""
    if not hyp_cues or hyp_cues[0][1] == 0: return None
    d = [min(abs(a - h[0]) for h in hyp_cues) for a, _, _ in ref_cues]
    return sum(d) / len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("srt")
    ap.add_argument("--start", type=float, default=0); ap.add_argument("--dur", type=float, default=600)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")

    clip = cut_clip(a.audio, a.start, a.dur)
    ref_cues = [(s - a.start, e - a.start, t) for s, e, t in parse_srt(a.srt) if s >= a.start and e <= a.start + a.dur]
    ref = norm(" ".join(t for _, _, t in ref_cues))
    (OUT / "ref.txt").write_text(ref, encoding="utf-8")
    print(f"ref: {len(ref_cues)} cues, {len(ref.split())} words")

    rows = []
    for m in a.models.split(","):
        slug = m.replace("/", "__")
        try:
            js, dt = transcribe(m, clip, key)
        except Exception as e:
            print(f"{m}: FAIL {e}"); rows.append((m, None, None, None, f"FAIL {str(e)[:60]}")); continue
        (OUT / f"{slug}.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
        cues = words_or_segments(js)
        write_srt(cues, OUT / f"{slug}.srt")
        hyp = norm(js.get("text") or " ".join(t for _, _, t in cues))
        off = timing_offset(ref_cues, cues)
        note = f"{dt:.0f}s, {len(cues)} cues" + (", REPEAT-LOOP" if repeat_flag(hyp) else "") + (f", offset {off:.2f}s" if off is not None else ", no timestamps")
        rows.append((m, wer(ref, hyp), cer(ref, hyp), dt, note))
        print(f"{m}: WER {rows[-1][1]:.1%} CER {rows[-1][2]:.1%} {note}")

    rows.sort(key=lambda r: (r[1] is None, r[1] or 0))
    md = "| model | WER | CER | notes |\n|---|---|---|---|\n" + "\n".join(
        f"| {m} | {'-' if w is None else f'{w:.1%}'} | {'-' if c is None else f'{c:.1%}'} | {n} |" for m, w, c, _, n in rows)
    (OUT / "results.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
