#!/usr/bin/env python3
"""Translate Nepali SRT -> English SRT via OpenRouter chat model. Timestamps are kept; only text changes.

usage:  python3 translate.py episode.srt            # -> episode.en.srt
        python3 translate.py outputs/               # every .srt (skips *.en.srt and existing outputs)
        python3 translate.py ep.srt --model anthropic/claude-sonnet-5.5   # slower/pricier, best quality
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from env import load_env

load_env()
API = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.8-flash"
CLEAN_MODEL = "google/gemini-3.5-flash-lite"  # cleanup is mechanical (undo transliteration) -> cheaper model is enough
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
FORMAT = ("Input is a JSON object {\"context_before\": [...], \"lines\": [...]}. context_before is only for understanding; "
          "do not output it. Reply with ONLY a JSON array of strings, exactly one per entry in \"lines\", same order, same count.")
SYSTEM = ("You translate Nepali podcast subtitles into English. Speakers often mix Nepali and English; lines or phrases already in "
          "English stay as they are. Write simple, natural, spoken English that an ordinary person would say, not formal or literary. "
          "Turn Nepali fillers into natural English (e.g. 'हैन?' -> 'right?', 'अनि' -> 'and' / 'so'). Keep the speaker's meaning and tone, "
          "keep names of people, places and brands in standard English spelling, do not add or drop information, do not merge or split lines. "
          + FORMAT)
SYSTEM_CLEAN = ("These are subtitles of a Nepali podcast where speakers mix Nepali with English. Rewrite each line so that every English word "
                "or phrase is written in normal English Latin letters with correct spelling, including English words that were written in "
                "Devanagari transliteration (e.g. इन्टरप्रेनरशिप -> entrepreneurship, डिफरेन्ट -> different, इन्भल्भ -> involved, एन्टिटी -> entity, "
                "थ्रुबाट -> through, एन्जल इन्भेस्टर -> Angel Investor, प्रोजेक्ट -> project). Keep all Nepali words in Devanagari exactly as they "
                "are (including Nepali endings attached to English words, e.g. 'involved छु'). Do not translate Nepali, do not add, remove, reorder or "
                "correct words, keep punctuation, and keep lines that need no change identical. " + FORMAT)


def parse_srt(path):
    from bakeoff import parse_srt as p
    return p(path)


def chat(model, payload, key, system=SYSTEM):
    err = ""
    for attempt in range(5):
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {key}"}, timeout=180, json={
                "model": model, "temperature": 0.2,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]})
            if r.status_code == 200:
                return r.json()
            err = f"{r.status_code} {r.text[:200]}"
            if r.status_code in (400, 401, 402, 403): raise SystemExit(f"fatal: {err}")
        except requests.RequestException as e:
            err = str(e)
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(err)


def parse_array(text):
    m = re.search(r"\[.*\]", text, re.S)
    arr = json.loads(m.group(0)) if m else None
    return [str(x).strip() for x in arr] if isinstance(arr, list) else None


def translate_batch(model, key, lines, before, system=SYSTEM):
    js = chat(model, {"context_before": before, "lines": lines}, key, system)
    cost = float((js.get("usage") or {}).get("cost") or 0)
    try:
        out = parse_array(js["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError):
        out = None
    if out is not None and len(out) == len(lines):
        return out, cost
    if len(lines) == 1:  # last resort: take raw text
        return [js["choices"][0]["message"]["content"].strip()], cost
    mid = len(lines) // 2  # count mismatch -> split and retry, keeps alignment
    a, ca = translate_batch(model, key, lines[:mid], before, system)
    b, cb = translate_batch(model, key, lines[mid:], lines[max(mid - 2, 0):mid], system)
    return a + b, cost + ca + cb


def translate_srt(src, dst=None, model=DEFAULT_MODEL, key=None, batch=40, workers=4, log=print, progress=None, system=SYSTEM):
    key = key or os.environ.get("OPENROUTER_API_KEY") or sys.exit("set OPENROUTER_API_KEY")
    from transcribe import write_srt
    cues = parse_srt(src)
    dst = Path(dst) if dst else Path(src).with_suffix(".en.srt")
    texts = [c[2] for c in cues]
    # cost saver: lines with no Devanagari are already English -> nothing to clean/translate, never sent to the model
    todo = [i for i, t in enumerate(texts) if DEVANAGARI.search(t)]
    jobs = [(todo[k:k + batch], [texts[i] for i in todo[k:k + batch]], texts[max(todo[k] - 3, 0):todo[k]])
            for k in range(0, len(todo), batch)]
    log(f"{'Cleaning' if system is SYSTEM_CLEAN else 'Translating'} {len(todo)} of {len(cues)} cues "
        f"({len(cues) - len(todo)} already English, skipped) in {len(jobs)} batches with {model}")
    done = [0]

    def run(j):
        r = translate_batch(model, key, j[1], j[2], system)
        done[0] += 1
        if progress: progress(done[0], len(jobs))
        return r
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(run, jobs))
    en = list(texts)
    for (idx, _, _), (out, _) in zip(jobs, results):
        for i, t in zip(idx, out): en[i] = t
    cost = sum(c for _, c in results)
    assert len(en) == len(cues), "translation count mismatch"
    write_srt([[t, c[0], c[1]] for t, c in zip(en, cues)], dst)
    log(f"  -> {dst} | cost ${cost:.3f}")
    return dst, cost


def clean_srt(path, model=CLEAN_MODEL, **kw):
    """Rewrite English words (even Devanagari-transliterated ones) in Latin letters. In place; raw copy kept as .raw.srt."""
    path = Path(path); raw = path.with_name(path.stem + ".raw.srt")
    if not raw.exists(): raw.write_bytes(path.read_bytes())
    return translate_srt(raw, dst=path, model=model, system=SYSTEM_CLEAN, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("--model", default=DEFAULT_MODEL); ap.add_argument("--force", action="store_true")
    ap.add_argument("--clean", action="store_true", help="fix English words written in Devanagari (in place, keeps .raw.srt)")
    a = ap.parse_args()
    p = Path(a.input)
    files = sorted(f for f in p.rglob("*.srt") if not f.name.endswith((".en.srt", ".raw.srt"))) if p.is_dir() else [p]
    total = 0
    if a.clean:
        for f in files:
            if not f.name.endswith((".raw.srt", ".en.srt")): total += clean_srt(f, model=a.model if a.model != DEFAULT_MODEL else CLEAN_MODEL)[1]
        return print(f"done | cost ${total:.3f}")
    for f in files:
        dst = f.with_suffix(".en.srt")
        if dst.exists() and not a.force: print(f"skip {f.name}"); continue
        total += translate_srt(f, dst, a.model)[1]
    print(f"done | cost ${total:.3f}")


if __name__ == "__main__":
    main()
