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
        if cur: cues.append([" ".join(w[0] for w in cur), cur[0][1], cur[-1][2], spk(cur[0])]); cur.clear()

    def spk(w): return w[3] if len(w) > 3 else None  # words carry a speaker name only when diarized

    for w in words:
        if cur:
            n = len(" ".join(x[0] for x in cur))
            if w[1] - cur[-1][2] > PAUSE_BREAK or n + 1 + len(w[0]) > MAX_CHARS or w[2] - cur[0][1] > MAX_DUR \
                    or spk(w) != spk(cur[-1]):
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
                and not out[-1][0].endswith(SENT_END[:3]) and out[-1][3] == c[3]:
            out[-1][0] += " " + c[0]; out[-1][2] = c[2]
        else:
            out.append(c)
    for i, c in enumerate(out):  # timing polish
        nxt = out[i + 1][1] if i + 1 < len(out) else float("inf")
        c[2] = min(max(c[2], c[1] + MIN_DUR), nxt - 0.04, c[1] + MAX_DUR + 1)
        c[2] = max(c[2], c[1] + 0.3)
    for p, c in zip(out, out[1:]):  # cues must never overlap: push a start that begins before the previous cue ends
        if c[1] < p[2]:
            c[1] = p[2]; c[2] = max(c[2], c[1] + 0.3)
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
    """cues: [text, start, end] or [text, start, end, speaker]; a speaker becomes a 'Name: ' prefix."""
    body = "".join(f"{i}\n{ts(c[1])} --> {ts(c[2])}\n{wrap(f'{c[3]}: {c[0]}' if len(c) > 3 and c[3] else c[0])}\n\n"
                   for i, c in enumerate(cues, 1))
    tmp = Path(str(path) + ".tmp"); tmp.write_text(body, encoding="utf-8"); tmp.replace(path)


def qa(cues, words):
    issues, w = [], [x[0] for x in words]
    for i in range(len(w) - 11):
        if w[i:i + 4] == w[i + 4:i + 8] == w[i + 8:i + 12]:
            issues.append(f"repeat-loop near '{' '.join(w[i:i+4])}'"); break
    hi = sum(1 for t, a, b, *_ in cues if len(t) / max(b - a, 0.1) > 25)
    if hi: issues.append(f"{hi} cues with reading speed >25 cps")
    return issues


def assign_speakers(words, turns):
    """Tag each word with the diarizer id of the turn containing its midpoint (else the nearest turn)."""
    turns = sorted(turns)
    if not turns: return words
    out = []  # overlapping turns: the shortest wins (interjections sit inside the other speaker's long turn)
    for w in words:
        mid = (w[1] + w[2]) / 2
        t = min((t for t in turns if t[0] <= mid <= t[1]), key=lambda t: t[1] - t[0], default=None) or min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))
        out.append((*w[:3], t[2]))
    return out


def _runs(words):
    """[(start_idx, end_idx_exclusive, speaker)] for consecutive words with the same speaker."""
    out, i = [], 0
    while i < len(words):
        j = i
        while j < len(words) and words[j][3] == words[i][3]: j += 1
        out.append((i, j, words[i][3])); i = j
    return out


def smooth_speakers(words, max_flip=2, snap=3):
    """Clean up speaker labels on the word stream.
    1) a run of <= max_flip words between two other runs joins the neighbour it sits closer to (kills 1-2 word flips caused
       by overlapping speech / interpolated word times);
    2) each remaining speaker change moves to the largest pause within `snap` words of it (people change turns in pauses)."""
    words = [list(w) for w in words]
    for _ in range(len(words)):  # each merge removes a run, so this terminates
        runs = _runs(words)
        k = next((k for k in range(1, len(runs) - 1) if runs[k][1] - runs[k][0] <= max_flip), None)
        if k is None: break
        a, b, _sp = runs[k]; prev, nxt = runs[k - 1], runs[k + 1]
        before, after = words[a][1] - words[a - 1][2], words[b][1] - words[b - 1][2]
        to = prev[2] if prev[2] == nxt[2] or before <= after else nxt[2]
        for i in range(a, b): words[i][3] = to
    runs = _runs(words)
    for k in range(1, len(runs)):
        b = runs[k][0]; lo = runs[k - 1][0] + 1; hi = runs[k][1] - 1  # the boundary may move inside the two neighbouring runs
        gap = lambda j: words[j][1] - words[j - 1][2]
        best = max(range(max(lo, b - snap), min(hi, b + snap) + 1), key=gap, default=b)
        if best != b and gap(best) > gap(b) + 0.1:
            if best < b:
                for i in range(best, b): words[i][3] = runs[k][2]
            else:
                for i in range(b, best): words[i][3] = runs[k - 1][2]
            runs = _runs(words)
    return [tuple(w) for w in words]


NAME_SYSTEM = ("You get the start of a transcript where each line is '[SPEAKER_xx] text' (ids come from a voice-separation tool and are "
               "reliable; names are not known). Work out each speaker's real name from what people say (self-introductions like 'I'm X', "
               "introductions like 'with me today is Y', being addressed by name). A cold-open teaser before the intro is usually a clip "
               "of the guest. The person who answers right after the host introduces them (e.g. 'thanks for having me') is that guest. "
               "Every id must get a distinct real name when the transcript or hint names allow it. Reply with ONLY a JSON object mapping every id to a full name. Use the optional hint names if they fit; if a "
               "speaker's name cannot be found, use 'Speaker N'.")


def name_speakers(cues, key, hints):
    """Replace diarizer ids in cue[3] with real names, inferred from the transcript by one cheap LLM call."""
    ids = []
    for c in cues:
        if c[3] not in ids: ids.append(c[3])
    label = {s: hints[i] if i < len(hints) else f"Speaker {i + 1}" for i, s in enumerate(ids)}  # fallback: first appearance
    try:
        import translate as TR
        sample = "\n".join(f"[{c[3]}] {c[0]}" for c in cues[:120])
        js = TR.chat(TR.CLEAN_MODEL, {"hint_names": hints, "speaker_ids": ids, "transcript": sample}, key, NAME_SYSTEM)
        state["cost"] += float((js.get("usage") or {}).get("cost") or 0)
        m = re.search(r"\{.*\}", js["choices"][0]["message"]["content"], re.S)
        got = json.loads(m.group(0)) if m else {}
        label.update({k: str(v).strip() for k, v in got.items() if k in label and str(v).strip()})
        unnamed = [k for k, v in label.items() if re.fullmatch(r"Speaker \d+", v)]
        spare = [h for h in hints if h not in label.values()]
        if len(unnamed) == 1 and len(spare) == 1: label[unnamed[0]] = spare[0]  # one hint name left for the one unnamed voice
        log("  speakers: " + ", ".join(f"{k} = {v}" for k, v in label.items()))
    except (Exception, SystemExit) as e:
        log(f"  could not name speakers from transcript ({str(e)[:100]}); using first-appearance order")
    for c in cues: c[3] = label[c[3]]
    return cues


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
    spk = getattr(a, "speakers", None)  # None = off; "" = on, names unknown; "A,B" = on, with hint names
    hints = [n.strip() for n in (spk or "").split(",") if n.strip()]
    dia = None
    if spk is not None:  # diarize on this machine WHILE the chunks are transcribed on OpenRouter, so the waits overlap
        import diarize
        log(f"  diarizing in background ({len(hints) or 'auto'} speakers)...")
        dia = ThreadPoolExecutor(1).submit(diarize.diarize, src, start, end, cache, len(hints) or None, log)
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
    if dia is not None:
        log("  waiting for diarization...")
        words = smooth_speakers(assign_speakers(words, dia.result()))
    cues = build_cues(words)
    if spk is not None and cues: cues = name_speakers(cues, key, hints)
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
    ap.add_argument("--speakers", nargs="?", const="", help='label speakers with pyannote; optional comma-separated hint names, e.g. "Host,Guest"')
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")
    p = Path(a.input)
    files = sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    g = load_glossary(a.glossary); ok = 0
    for f in files:
        srt = process(f, a, key, g)
        if not srt: continue
        ok += 1
    log(f"done {ok}/{len(files)} | total cost ${state['cost']:.3f}")


if __name__ == "__main__":
    main()
