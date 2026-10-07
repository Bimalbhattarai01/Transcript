#!/usr/bin/env python3
"""Nepali podcast -> SRT via OpenRouter STT (default openai/gpt-transcribe).

Why chunk-based: gpt-transcribe may not return timestamps and OpenRouter times out at ~60s per call.
So audio is cut at real silences into short chunks (<= --max-chunk s). Each chunk is transcribed alone,
which also stops Whisper-style drift/hallucination across long audio. Word times come from the API if
it returns them, else are interpolated over the chunk's *speech-only* timeline (pauses get no text).
Words are then packed into subtitle cues (<=2 lines x 42 chars, <=6.5s, split on pauses / danda).

usage:
  export OPENROUTER_API_KEY=...
  python3 transcribe.py episode.mp3 --start 300 --limit-seconds 600 --compare capcut.srt   # test
  python3 transcribe.py /path/to/episodes --workers 6                                      # batch
Resumable: raw chunk JSON cached in <out>/.cache/<episode>/ ; existing .srt are skipped (--force).
"""
import argparse, base64, json, os, random, re, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from env import load_env

load_env()
API = "https://openrouter.ai/api/v1/audio/transcriptions"
EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".flac", ".ogg", ".aac", ".webm", ".mov"}
LINE_CHARS, MAX_CHARS, MAX_DUR, MIN_DUR, PAUSE_BREAK = 42, 84, 6.5, 1.0, 0.6
SENT_END = ("।", "?", "!", ".", "॥")

hooks = {"log": print, "progress": lambda done, total: None}


def log(msg):
    hooks["log"](msg)


state = {"verbose": True, "cost": 0.0, "lock": threading.Lock()}


# ---------- audio ----------
def probe_duration(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def speech_spans(src, start, end, noise_db, min_sil):
    """Complement of ffmpeg silencedetect inside [start, end]."""
    cmd = ["ffmpeg", "-nostats", "-ss", str(start), "-t", str(end - start), "-i", str(src), "-vn",
           "-af", f"silencedetect=noise={noise_db}dB:d={min_sil}", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True).stderr
    sil, cur = [], None
    for line in err.splitlines():
        if m := re.search(r"silence_start: (-?[\d.]+)", line):
            cur = max(float(m[1]), 0) + start
        elif (m := re.search(r"silence_end: (-?[\d.]+)", line)) and cur is not None:
            sil.append((cur, float(m[1]) + start)); cur = None
    if cur is not None:
        sil.append((cur, end))
    spans, prev = [], start
    for s, e in sil:
        if s - prev > 0.15: spans.append((prev, s))
        prev = max(prev, e)
    if end - prev > 0.15: spans.append((prev, end))
    return spans


def make_chunks(spans, max_chunk, soft_chunk):
    """Group speech spans into chunks <= max_chunk, preferring to close after a >=0.8s pause once >= soft_chunk."""
    pieces = []
    for s, e in spans:  # split over-long spans (continuous speech/music)
        n = max(1, -(-int((e - s) * 1000) // int(max_chunk * 1000)))
        step = (e - s) / n
        pieces += [(s + i * step, s + (i + 1) * step) for i in range(n)]
    chunks, cur = [], []
    for p in pieces:
        if cur and (p[1] - cur[0][0] > max_chunk or (cur[-1][1] - cur[0][0] >= soft_chunk and p[0] - cur[-1][1] >= 0.8)):
            chunks.append(cur); cur = []
        cur.append(p)
    if cur: chunks.append(cur)
    return [c for c in chunks if sum(e - s for s, e in c) >= 0.4]


def cut_audio(src, start, end):
    pad_s, pad_e = max(start - 0.1, 0), end + 0.1
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{pad_s:.3f}", "-t", f"{pad_e - pad_s:.3f}", "-i", str(src),
                           "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", "-f", "mp3", "pipe:1"],
                          capture_output=True, check=True).stdout


# ---------- api ----------
def call_api(model, mp3, key, lang):
    for attempt in range(6):
        body = {"model": model, "language": lang, "input_audio": {"data": base64.b64encode(mp3).decode(), "format": "mp3"}}
        used_verbose = state["verbose"]  # snapshot: parallel workers may flip the flag mid-flight
        if used_verbose:
            body.update(response_format="verbose_json", timestamp_granularities=["word"])
        try:
            r = requests.post(API, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        except requests.RequestException as e:
            r = None; err = str(e)
        else:
            if r.status_code == 200: return r.json()
            err = f"{r.status_code} {r.text[:200]}"
            if r.status_code == 400 and used_verbose:
                if state["verbose"]: log("  (model rejects verbose_json -> using plain json, interpolated timing)")
                state["verbose"] = False; continue
            if r.status_code in (401, 402, 403) or (r.status_code == 400): raise SystemExit(f"fatal: {err}")
        time.sleep(min(2 ** attempt, 30) + random.random())
    raise RuntimeError(err)


def transcribe_chunk(src, ch, idx, cache, model, key, lang):
    f = cache / f"{idx:05d}_{int(ch[0][0]*1000)}_{int(ch[-1][1]*1000)}.json"
    if f.exists(): return json.loads(f.read_text(encoding="utf-8"))
    js = call_api(model, cut_audio(src, ch[0][0], ch[-1][1]), key, lang)
    with state["lock"]:
        state["cost"] += float((js.get("usage") or {}).get("cost") or 0)
    f.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    return js


# ---------- text -> timed words ----------
def to_wall(x, spans):
    """Map position x on the speech-only timeline to wall-clock time."""
    acc = 0.0
    for s, e in spans:
        if x <= acc + (e - s): return s + (x - acc)
        acc += e - s
    return spans[-1][1]


def timed_words(js, spans, glossary):
    text = js.get("text", "").strip()
    for a, b in glossary: text = text.replace(a, b)
    if not text: return []
    api_words = js.get("words") or []
    t0 = spans[0][0] - 0.0  # chunk audio starts 0.1s before first span (clamped at 0)
    if api_words and all("start" in w for w in api_words):
        off = max(t0 - 0.1, 0)
        out = [((w.get("word") or w.get("text", "")).strip(), off + w["start"], off + w["end"]) for w in api_words]
        out = [w for w in out if w[0]]
        if out:
            for a, b in glossary: out = [(w[0].replace(a, b), w[1], w[2]) for w in out]
            return out
    words = text.split()
    wts = [len(w) + 1 for w in words]; tot = sum(wts)
    speech = sum(e - s for s, e in spans)
    out, acc = [], 0
    for w, wt in zip(words, wts):
        out.append((w, to_wall(acc / tot * speech, spans), to_wall((acc + wt) / tot * speech - 0.001, spans)))
        acc += wt
    return out


# ---------- cues ----------
def build_cues(words):
    cues, cur = [], []

    def flush():
        if cur: cues.append([" ".join(w[0] for w in cur), cur[0][1], cur[-1][2]]); cur.clear()

    for w in words:
        if cur:
            n = len(" ".join(x[0] for x in cur))
            if w[1] - cur[-1][2] > PAUSE_BREAK or n + 1 + len(w[0]) > MAX_CHARS or w[2] - cur[0][1] > MAX_DUR:
                flush()
        cur.append(w)
        n = len(" ".join(x[0] for x in cur))
        if (w[0].endswith(SENT_END) and n >= 14) or (w[0].endswith((",", "–", "—")) and n >= 55):
            flush()
    flush()
    # merge too-short cues into a neighbour when it fits
    out = []
    for c in cues:
        if out and (out[-1][2] - out[-1][1] < MIN_DUR or c[2] - c[1] < MIN_DUR) \
                and c[1] - out[-1][2] < 0.4 and len(out[-1][0]) + 1 + len(c[0]) <= MAX_CHARS and c[2] - out[-1][1] <= MAX_DUR \
                and not out[-1][0].endswith(SENT_END[:3]):
            out[-1][0] += " " + c[0]; out[-1][2] = c[2]
        else:
            out.append(c)
    for i, c in enumerate(out):  # timing polish
        nxt = out[i + 1][1] if i + 1 < len(out) else float("inf")
        c[2] = min(max(c[2], c[1] + MIN_DUR), nxt - 0.04, c[1] + MAX_DUR + 1)
        c[2] = max(c[2], c[1] + 0.3)
    return out


def wrap(text):
    if len(text) <= LINE_CHARS: return text
    mid = len(text) / 2
    sp = [i for i, ch in enumerate(text) if ch == " "]
    if not sp: return text
    i = min(sp, key=lambda k: abs(k - mid))
    return text[:i] + "\n" + text[i + 1:]


def ts(x):
    ms = round(max(x, 0) * 1000)
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def write_srt(cues, path):
    body = "".join(f"{i}\n{ts(a)} --> {ts(b)}\n{wrap(t)}\n\n" for i, (t, a, b) in enumerate(cues, 1))
    tmp = Path(str(path) + ".tmp"); tmp.write_text(body, encoding="utf-8"); tmp.replace(path)


def qa(cues, words):
    issues, w = [], [x[0] for x in words]
    for i in range(len(w) - 11):
        if w[i:i + 4] == w[i + 4:i + 8] == w[i + 8:i + 12]:
            issues.append(f"repeat-loop near '{' '.join(w[i:i+4])}'"); break
    hi = sum(1 for t, a, b in cues if len(t) / max(b - a, 0.1) > 25)
    if hi: issues.append(f"{hi} cues with reading speed >25 cps")
    return issues


# ---------- orchestration ----------
def load_glossary(p):
    if not p: return []
    return [tuple(l.rstrip("\n").split("\t", 1)) for l in Path(p).read_text(encoding="utf-8").splitlines() if "\t" in l]


def process(src, a, key, glossary):
    src = Path(src); out_dir = Path(a.out) if a.out else src.parent; out_dir.mkdir(parents=True, exist_ok=True)
    limited = a.limit_seconds is not None
    tag = getattr(a, "tag", "")  # per-model suffix so several models can run on one episode
    srt_path = out_dir / (src.stem + (f".{tag}" if tag else "") + (".test" if limited else "") + ".srt")
    if srt_path.exists() and not a.force:
        log(f"skip {src.name} (exists)"); return srt_path
    dur = probe_duration(src)
    start = a.start; end = min(dur, start + a.limit_seconds) if limited else dur
    spans = speech_spans(src, start, end, a.noise_db, a.min_silence)
    chunks = make_chunks(spans, a.max_chunk, a.soft_chunk)
    cache = out_dir / ".cache" / (src.stem + (f"-{tag}" if tag else "")); cache.mkdir(parents=True, exist_ok=True)
    log(f"{src.name}: {end-start:.0f}s audio, {len(chunks)} chunks")
    c0 = state["cost"]; t0 = time.time()
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            done = [0]

            def work(ic):
                r = transcribe_chunk(src, ic[1], ic[0], cache, a.model, key, a.lang)
                done[0] += 1; hooks["progress"](done[0], len(chunks))
                return r
            results = list(ex.map(work, enumerate(chunks)))
    except Exception as e:
        log(f"FAILED {src.name}: {e}"); Path("failures.txt").open("a").write(f"{src}\t{e}\n"); return False
    words = [w for js, ch in zip(results, chunks) for w in timed_words(js, ch, glossary)]
    cues = build_cues(words)
    write_srt(cues, srt_path)
    log(f"  -> {srt_path} | {len(cues)} cues | {time.time()-t0:.0f}s | cost ${state['cost']-c0:.3f}")
    for i in qa(cues, words): log(f"  QA: {i}")
    if a.compare:
        sys.path.insert(0, str(Path(__file__).parent))
        import bakeoff as b
        ref = [(s, e, t) for s, e, t in b.parse_srt(a.compare) if s >= start and e <= end]
        r = b.norm(" ".join(t for _, _, t in ref)); h = b.norm(" ".join(c[0] for c in cues))
        log(f"  vs CapCut: WER {b.wer(r, h):.1%} CER {b.cer(r, h):.1%} ({len(r.split())} ref words, {len(ref)} ref cues vs {len(cues)})")
    return srt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="audio/video file or folder")
    ap.add_argument("--out"); ap.add_argument("--model", default="openai/gpt-transcribe"); ap.add_argument("--lang", default="ne")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-chunk", type=float, default=20); ap.add_argument("--soft-chunk", type=float, default=10)
    ap.add_argument("--noise-db", type=float, default=-35); ap.add_argument("--min-silence", type=float, default=0.35)
    ap.add_argument("--start", type=float, default=0); ap.add_argument("--limit-seconds", type=float)
    ap.add_argument("--compare", help="CapCut .srt to score against"); ap.add_argument("--glossary", help="TSV: wrong<TAB>right")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True,
                    help="write English words in English letters (keeps .raw.srt)")
    ap.add_argument("--translate", action=argparse.BooleanOptionalAction, default=True, help="also write <name>.en.srt (English)")
    ap.add_argument("--tr-model", default=None, help="model for clean/translate (default: translate.DEFAULT_MODEL)")
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")
    p = Path(a.input)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    g = load_glossary(a.glossary); ok = 0
    import translate as TR  # lazy: translate imports this module
    tr_model = a.tr_model or TR.DEFAULT_MODEL
    for f in files:
        srt = process(f, a, key, g)
        if not srt: continue
        ok += 1
        try:
            if a.clean and not Path(srt).with_name(Path(srt).stem + ".raw.srt").exists():
                state["cost"] += TR.clean_srt(srt, model=a.tr_model or TR.CLEAN_MODEL, key=key, log=log)[1]
            en = Path(srt).with_suffix(".en.srt")
            if a.translate and (a.force or not en.exists()):
                state["cost"] += TR.translate_srt(srt, en, model=tr_model, key=key, log=log)[1]
        except (Exception, SystemExit) as e:
            log(f"  clean/translate failed for {Path(srt).name}: {e} (Nepali SRT kept)")
    log(f"done {ok}/{len(files)} | total cost ${state['cost']:.3f}")


if __name__ == "__main__":
    main()
