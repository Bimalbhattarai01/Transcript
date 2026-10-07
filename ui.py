#!/usr/bin/env python3
"""Local web UI: paste a YouTube link -> download audio (yt-dlp) -> transcribe with one or MORE models -> SRT
(optionally translated to English), with live progress, prices and a side-by-side model comparison.

run:  python3 ui.py     then open http://127.0.0.1:8765
"""
import json, os, re, subprocess, threading, time, uuid, webbrowser
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
import transcribe as T
import translate as TR

ROOT = Path(__file__).parent
DL, OUT = ROOT / "downloads", ROOT / "outputs"
jobs, run_lock = {}, threading.Lock()  # one job at a time (transcribe hooks are global)
EPISODE_TOKENS = (40_000, 30_000)  # rough in/out tokens to translate a ~1.5h episode
TR_CHOICES = ["google/gemini-3.8-flash", "google/gemini-3.5-flash-lite", "openai/gpt-5.6-luna", "openai/gpt-5.6-terra",
              "anthropic/claude-sonnet-5.5"]
DEFAULT_STT = ["openai/gpt-transcribe"]
_models = {"t": 0, "data": None}


def add(job, msg):
    job["log"].append(msg)


# ---------- model list with prices (live from OpenRouter, no key needed) ----------
def stt_price(m):
    p = float(m["pricing"]["prompt"]); c = float(m["pricing"].get("completion") or 0)
    if c > 0: return f"token-based: ${p*1e6:.2f}/M in, ${c*1e6:.2f}/M out"
    if p < 0.01: return f"${p*3600:.2f}/hour"  # per-second pricing (matches vendor list prices for gpt-transcribe, chirp-3, nova-3)
    return f"${p:g} (unit unclear, check actual cost after run)"


def get_models():
    if _models["data"] and time.time() - _models["t"] < 600: return _models["data"]
    stt, tr = [], []
    try:
        base = "https://openrouter.ai/api/v1/models"
        for m in requests.get(base + "?output_modalities=transcription", timeout=15).json()["data"]:
            stt.append({"id": m["id"], "name": m["name"], "price": stt_price(m)})
        by_id = {m["id"]: m for m in requests.get(base, timeout=15).json()["data"]}
        for i in TR_CHOICES:
            if i in by_id:
                pi, po = float(by_id[i]["pricing"]["prompt"]), float(by_id[i]["pricing"]["completion"])
                est = pi * EPISODE_TOKENS[0] + po * EPISODE_TOKENS[1]
                tr.append({"id": i, "name": by_id[i]["name"], "price": f"${pi*1e6:.2f}/M in, ${po*1e6:.2f}/M out", "est": f"~${est:.2f}/episode"})
    except Exception as e:
        print("model list failed:", e)
    if not stt: stt = [{"id": i, "name": i, "price": "?"} for i in DEFAULT_STT]
    if not tr: tr = [{"id": TR.DEFAULT_MODEL, "name": TR.DEFAULT_MODEL, "price": "?", "est": ""}]
    _models.update(t=time.time(), data={"stt": stt, "tr": tr, "default_stt": DEFAULT_STT, "default_tr": TR.DEFAULT_MODEL})
    return _models["data"]


# ---------- jobs ----------
def do_translate(job, r, model):
    job.update(stage="translate", progress=0, detail=f"{r['model']} -> English")
    dst, c = TR.translate_srt(r["srt"], model=model, key=os.environ.get("OPENROUTER_API_KEY"), log=lambda m: add(job, m),
                              progress=lambda d, t: job.update(progress=100 * d / t, detail=f"English {d}/{t} batches"))
    r.update(en_srt=str(dst), en_text=Path(dst).read_text(encoding="utf-8"), en_model=model, en_cost=round(c, 4))


def run_translate(job, r, model):
    with run_lock:
        try:
            do_translate(job, r, model)
        except (Exception, SystemExit) as e:
            add(job, f"TRANSLATE ERROR: {e}")
        finally:
            job.update(stage="done", progress=100, status="done")


def download(job, url):
    job.update(stage="download", progress=0)
    add(job, "Downloading audio with yt-dlp...")
    ytdlp = str(ROOT / "bin" / "yt-dlp") if (ROOT / "bin" / "yt-dlp").exists() else "yt-dlp"
    # YouTube needs a JS runtime (node) + fresh yt-dlp, else 403
    p = subprocess.Popen([ytdlp, "--js-runtimes", "node", "--remote-components", "ejs:github", "--no-playlist", "--newline",
                          "-f", "bestaudio", "-x", "--audio-format", "mp3", "--print", "after_move:filepath",
                          "-o", str(DL / "%(id)s.%(ext)s"), url], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    path = None
    for line in p.stdout:
        line = line.strip()
        if m := re.search(r"\[download\]\s+([\d.]+)%", line):
            job["progress"] = float(m[1])
        elif line.startswith("/") and line.endswith(".mp3"):
            path = line
        elif line:
            add(job, line)
    if p.wait() != 0 or not path: raise RuntimeError("yt-dlp failed (see log)")
    add(job, f"Downloaded {Path(path).name}")
    return path


def run_job(job, url, start, limit, models, want_en, en_model, clean=False):
    with run_lock:
        try:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key: raise RuntimeError("OPENROUTER_API_KEY missing in .env")
            DL.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
            path = download(job, url)
            T.hooks["log"] = lambda m: add(job, m)
            for n, model in enumerate(models, 1):
                job.update(stage="transcribe", progress=0, detail=f"{model} ({n}/{len(models)})")
                add(job, f"=== {model} ===")
                T.hooks["progress"] = lambda d, t, n=n, model=model: job.update(
                    progress=100 * d / t, detail=f"{model} ({n}/{len(models)}) {d}/{t} chunks")
                tag = model.replace("/", "_")
                a = Namespace(out=str(OUT), model=model, lang="ne", workers=6, max_chunk=20, soft_chunk=10, noise_db=-35,
                              min_silence=0.35, start=start, limit_seconds=limit, compare=None, glossary=None, force=True, tag=tag)
                c0, t0 = T.state["cost"], time.time()
                try:
                    if not T.process(path, a, key, []): raise RuntimeError("transcription failed (see log)")
                except (Exception, SystemExit) as e:
                    add(job, f"{model} FAILED: {e}")
                    job["results"].append({"model": model, "error": str(e)[:200]}); continue
                srt = OUT / (Path(path).stem + f".{tag}" + (".test" if limit else "") + ".srt")
                text = srt.read_text(encoding="utf-8")
                r = {"model": model, "srt": str(srt), "text": text, "cost": round(T.state["cost"] - c0, 4),
                     "seconds": round(time.time() - t0), "cues": text.count("-->"), "words": len(text.split())}
                job["results"].append(r)
                if clean:
                    try:
                        job.update(stage="translate", progress=0, detail=f"{model}: English words in English letters")
                        _, cc = TR.clean_srt(srt, model=TR.CLEAN_MODEL, key=key, log=lambda m: add(job, m),
                                             progress=lambda d, t: job.update(progress=100 * d / t))
                        r.update(text=srt.read_text(encoding="utf-8"), cost=round(r["cost"] + cc, 4))
                    except (Exception, SystemExit) as e:
                        add(job, f"CLEANUP ERROR: {e} (raw transcript kept)")
                if want_en:
                    try:
                        do_translate(job, r, en_model)
                    except (Exception, SystemExit) as e:
                        add(job, f"TRANSLATE ERROR: {e} (Nepali SRT is still saved)")
            ok = any("srt" in r for r in job["results"])
            job.update(stage="done" if ok else "error", status="done" if ok else "error", progress=100)
        except (Exception, SystemExit) as e:  # SystemExit: fatal API errors (bad key, no credits)
            add(job, f"ERROR: {e}"); job.update(status="error", stage="error")
        finally:
            T.hooks["log"] = print; T.hooks["progress"] = lambda d, t: None


def run_existing(job, path, clean, want_en, en_model):
    """Clean and/or translate an SRT that is already in outputs/ (no download, no transcription cost)."""
    path = Path(path)
    r = {"model": path.name, "srt": str(path), "cost": 0.0, "seconds": 0}
    with run_lock:
        try:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key: raise RuntimeError("OPENROUTER_API_KEY missing in .env")
            job["results"].append(r)
            t0 = time.time()
            if clean:
                job.update(stage="translate", progress=0, detail="English words in English letters")
                _, c = TR.clean_srt(path, model=TR.CLEAN_MODEL, key=key, log=lambda m: add(job, m),
                                    progress=lambda d, t: job.update(progress=100 * d / t, detail=f"cleanup {d}/{t} batches"))
                r["cost"] = round(r["cost"] + c, 4)
            text = path.read_text(encoding="utf-8")
            r.update(text=text, cues=text.count("-->"), words=len(text.split()))
            if want_en:
                do_translate(job, r, en_model)
                r["cost"] = round(r["cost"] + r.get("en_cost", 0), 4)
            r["seconds"] = round(time.time() - t0)
            job.update(stage="done", status="done", progress=100)
        except (Exception, SystemExit) as e:
            add(job, f"ERROR: {e}")
            if "text" in r:
                job.update(stage="done", status="done")  # partial result still usable
            else:
                if r in job["results"]: job["results"].remove(r)
                job.update(status="error", stage="error")


def list_srts():
    out = []
    files = sorted(OUT.glob("*.srt"), key=lambda f: -f.stat().st_mtime) if OUT.exists() else []
    for f in files:
        if f.name.endswith((".en.srt", ".raw.srt")): continue
        t = f.read_text(encoding="utf-8")
        dev = sum(1 for blk in t.split("\n\n") if any("\u0900" <= ch <= "\u097f" for ch in blk))
        out.append({"name": f.name, "cues": t.count("-->"), "nepali_cues": dev, "has_en": f.with_suffix(".en.srt").exists()})
    return out


def result_meta(r):
    return {k: v for k, v in r.items() if k not in ("text", "en_text")} | {"has_en": "en_text" in r}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json", extra=None):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); parts = u.path.strip("/").split("/")
        if u.path == "/": return self.send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/srts": return self.send(200, json.dumps(list_srts()))
        if u.path == "/api/models": return self.send(200, json.dumps(get_models()))
        if parts[:2] == ["api", "job"] and len(parts) == 3 and parts[2] in jobs:
            j = jobs[parts[2]]; frm = int(parse_qs(u.query).get("from", ["0"])[0])
            body = {k: v for k, v in j.items() if k not in ("log", "results")}
            return self.send(200, json.dumps({**body, "log": j["log"][frm:], "next": len(j["log"]),
                                              "results": [result_meta(r) for r in j["results"]]}))
        if parts[:2] == ["api", "result"] and len(parts) == 4 and parts[2] in jobs:  # text of one result
            try: r = jobs[parts[2]]["results"][int(parts[3])]
            except (ValueError, IndexError): return self.send(404, "{}")
            return self.send(200, json.dumps({"text": r.get("text", ""), "en_text": r.get("en_text", "")}))
        if parts[:2] == ["api", "file"] and len(parts) == 4 and parts[2] in jobs:
            try: r = jobs[parts[2]]["results"][int(parts[3])]
            except (ValueError, IndexError): return self.send(404, "{}")
            en = bool(parse_qs(u.query).get("en")) and "en_text" in r
            return self.send(200, r["en_text"] if en else r["text"], "application/x-subrip; charset=utf-8",
                             {"Content-Disposition": f'attachment; filename="{Path(r["en_srt"] if en else r["srt"]).name}"'})
        self.send(404, "{}")

    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
        if self.path == "/api/translate":
            j = jobs.get(d.get("id"))
            try: r = j["results"][int(d.get("idx", 0))]
            except (TypeError, KeyError, ValueError, IndexError): r = None
            if not j or j.get("status") == "running" or not r or "srt" not in r:
                return self.send(400, json.dumps({"error": "no finished result"}))
            j.update(status="running", want_en=True)
            threading.Thread(target=run_translate, args=(j, r, d.get("en_model") or TR.DEFAULT_MODEL), daemon=True).start()
            return self.send(200, "{}")
        if self.path == "/api/srt-job":
            f = OUT / Path(d.get("file") or "").name
            if not f.is_file() or f.suffix != ".srt": return self.send(400, json.dumps({"error": "SRT not found in outputs/"}))
            if not (d.get("clean") or d.get("en")): return self.send(400, json.dumps({"error": "Tick cleanup and/or English"}))
            jid = uuid.uuid4().hex[:8]
            jobs[jid] = {"status": "running", "stage": "queued", "progress": 0, "log": [], "results": [], "want_en": True}
            threading.Thread(target=run_existing, args=(jobs[jid], f, bool(d.get("clean")), bool(d.get("en")),
                                                        d.get("en_model") or TR.DEFAULT_MODEL), daemon=True).start()
            return self.send(200, json.dumps({"id": jid}))
        if self.path != "/api/start": return self.send(404, "{}")
        url = (d.get("url") or "").strip()
        if not re.match(r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", url):
            return self.send(400, json.dumps({"error": "Paste a valid YouTube link"}))
        models = [m for m in (d.get("models") or []) if re.fullmatch(r"[\w.\-]+/[\w.\-:]+", m)]
        if not models: return self.send(400, json.dumps({"error": "Pick at least one transcription model"}))
        try:
            limit = float(d["limit"]) * 60 if d.get("limit") else None
            start = float(d.get("start") or 0) * 60
        except ValueError:
            return self.send(400, json.dumps({"error": "Start/limit must be numbers"}))
        jid = uuid.uuid4().hex[:8]
        jobs[jid] = {"status": "running", "stage": "queued", "progress": 0, "log": [], "results": [], "url": url,
                     "want_en": bool(d.get("en") or d.get("clean"))}
        threading.Thread(target=run_job, args=(jobs[jid], url, start, limit, models, bool(d.get("en")),
                                               d.get("en_model") or TR.DEFAULT_MODEL, bool(d.get("clean"))), daemon=True).start()
        self.send(200, json.dumps({"id": jid}))


PAGE = r"""<!doctype html><meta charset=utf-8><title>Nepali SRT</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}body{font:15px system-ui;background:#0f1115;color:#e6e6e6;max-width:940px;margin:32px auto;padding:0 16px}
h1{font-size:20px}h3{margin:18px 0 8px}input,button,select{font:inherit;padding:9px 12px;border-radius:8px;border:1px solid #333;background:#181b22;color:inherit}
input#url{width:100%;box-sizing:border-box;margin-bottom:10px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
input.n{width:90px}button{background:#3b6cf6;border:0;cursor:pointer}button:disabled{opacity:.5}
.models{border:1px solid #262a33;border-radius:8px;max-height:230px;overflow:auto;margin-bottom:12px}
.models label{display:flex;gap:10px;padding:7px 12px;border-bottom:1px solid #1d2028;cursor:pointer;align-items:baseline}
.models label:hover{background:#171a21}.models b{font-weight:600}.models span{color:#8a93a6;font-size:13px;margin-left:auto;text-align:right}
.steps{display:flex;gap:6px;margin:16px 0}.s{flex:1;padding:8px;border-radius:8px;background:#181b22;text-align:center;color:#777;font-size:13px}
.s.on{background:#23325e;color:#fff}.s.ok{background:#1d3b2a;color:#9be3b0}
.bar{height:8px;background:#181b22;border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;width:0;background:#3b6cf6;transition:width .3s}
pre{background:#0a0c10;border:1px solid #222;border-radius:8px;padding:12px;max-height:200px;overflow:auto;font-size:12px;white-space:pre-wrap}
textarea{width:100%;height:340px;box-sizing:border-box;background:#0a0c10;color:#e6e6e6;border:1px solid #222;border-radius:8px;padding:12px;font:14px/1.5 monospace}
.err{color:#ff8080}small,.muted{color:#8a93a6}a.btn{background:#1f9d55;color:#fff;padding:8px 12px;border-radius:8px;text-decoration:none;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:10px}td,th{padding:6px 8px;border-bottom:1px solid #222;text-align:left}
tr.pick{background:#1b2338;cursor:pointer}tr.pick:hover{background:#222c47}tr.sel{outline:1px solid #3b6cf6}
</style>
<h1>Nepali podcast &rarr; SRT</h1>
<input id=url placeholder="Paste YouTube link (https://youtu.be/...)">
<div class=row>
 <label>Start at (min) <input class=n id=start type=number min=0 placeholder=0></label>
 <label>Only first (min) <input class=n id=limit type=number min=1 placeholder="full"></label>
 <small>Tip: "Only first 10" = cheap test run.</small>
</div>
<h3>Transcription models <small id=nsel></small></h3>
<div class=models id=stt>loading models...</div>
<div class=row>
 <label><input type=checkbox id=clean checked> Write English words in English letters</label>
 <label><input type=checkbox id=en checked> Also translate to English</label> <small>using</small>
 <select id=enmodel></select><span class=muted id=enprice></span>
 <button id=go style="margin-left:auto">Generate SRT</button>
</div>
<details style="margin:10px 0"><summary class=muted style="cursor:pointer">Already have an SRT in outputs/? Clean or translate it (no transcription cost)</summary>
 <div class=row style="margin-top:10px"><select id=ex style="min-width:320px"></select>
  <label><input type=checkbox id=exclean checked> English words in English letters</label>
  <label><input type=checkbox id=exen checked> English SRT</label>
  <button id=exgo>Run on this SRT</button></div></details>
<div class=steps><div class=s id=s_download>1 Download</div><div class=s id=s_transcribe>2 Transcribe</div><div class=s id=s_translate>3 English</div><div class=s id=s_done>4 Done</div></div>
<div class=bar><i id=bar></i></div><small id=detail></small>
<h3>Log</h3><pre id=log></pre>
<div id=res style="display:none">
 <h3>Results <small>(click a row to view; compare cost, speed and read the text)</small></h3>
 <table id=tbl></table>
 <div class=row><b id=cur></b> <span style="margin-left:auto"></span>
  <button id=tabNe>Nepali</button><button id=tabEn>English</button>
  <a class=btn id=dl href=#>Download .srt</a><a class=btn id=dlEn href=# style="display:none">Download .en.srt</a>
  <button id=tr>Translate to English</button></div>
 <textarea id=txt readonly></textarea>
</div>
<script>
const $=id=>document.getElementById(id);let jid,from=0,timer,sel=0,mode='ne',cache={},M;
const q=a=>JSON.stringify(a);
async function init(){
 M=await (await fetch('/api/models')).json();
 $('stt').innerHTML=M.stt.map(m=>`<label><input type=checkbox value="${m.id}" ${M.default_stt.includes(m.id)?'checked':''}><b>${m.name}</b><small>${m.id}</small><span>${m.price}</span></label>`).join('');
 $('enmodel').innerHTML=M.tr.map(m=>`<option value="${m.id}">${m.name}</option>`).join('');
 $('enmodel').value=M.default_tr;upd();
 $('stt').onchange=upd;$('enmodel').onchange=upd;
}
function upd(){
 $('nsel').textContent='('+picked().length+' selected)';
 const m=M.tr.find(x=>x.id==$('enmodel').value);$('enprice').textContent=m?m.price+' · '+m.est:'';
}
const picked=()=>[...document.querySelectorAll('#stt input:checked')].map(x=>x.value);
$('go').onclick=async()=>{
 $('log').textContent='';$('res').style.display='none';from=0;cache={};sel=0;
 const r=await fetch('/api/start',{method:'POST',body:q({url:$('url').value,start:$('start').value,limit:$('limit').value,models:picked(),en:$('en').checked,clean:$('clean').checked,en_model:$('enmodel').value})});
 const d=await r.json();if(!r.ok){$('log').innerHTML='<span class=err>'+d.error+'</span>';return}
 jid=d.id;$('go').disabled=true;clearInterval(timer);timer=setInterval(poll,1000);
};
$('tr').onclick=async()=>{
 await fetch('/api/translate',{method:'POST',body:q({id:jid,idx:sel,en_model:$('enmodel').value})});
 $('go').disabled=true;clearInterval(timer);timer=setInterval(poll,1000);
};
$('tabNe').onclick=()=>{mode='ne';show()};$('tabEn').onclick=()=>{mode='en';show()};
async function poll(){
 const j=await (await fetch(`/api/job/${jid}?from=${from}`)).json();from=j.next;
 if(j.log.length){$('log').textContent+=j.log.join('\n')+'\n';$('log').scrollTop=1e9}
 $('bar').style.width=(j.progress||0)+'%';$('detail').textContent=(j.detail||'')+' '+Math.round(j.progress||0)+'%';
 $('s_translate').style.display=j.want_en?'':'none';
 const order=['download','transcribe','translate','done'],i=order.indexOf(j.stage);
 order.forEach((s,k)=>$('s_'+s).className='s'+(k<i||j.stage=='done'?' ok':k==i?' on':''));
 renderTable(j.results);
 if(j.status!='running'){clearInterval(timer);$('go').disabled=false;show()}
}
function renderTable(rs){
 if(!rs.length)return;$('res').style.display='block';
 $('tbl').innerHTML='<tr><th>Model</th><th>Cost</th><th>Time</th><th>Cues</th><th>Words</th><th>English</th></tr>'+rs.map((r,k)=>
  r.error?`<tr><td>${r.model}</td><td colspan=5 class=err>${r.error}</td></tr>`:
  `<tr class="pick ${k==sel?'sel':''}" onclick="sel=${k};show()"><td>${r.model}</td><td>$${r.cost}${r.en_cost?' + $'+r.en_cost:''}</td><td>${r.seconds}s</td><td>${r.cues}</td><td>${r.words}</td><td>${r.has_en?'yes':'-'}</td></tr>`).join('');
 window.RS=rs;
}
async function show(){
 const r=(window.RS||[])[sel];if(!r||r.error)return;
 $('cur').textContent=r.model;
 const key=sel+':'+r.has_en;
 if(!cache[key])cache[key]=await (await fetch(`/api/result/${jid}/${sel}`)).json();
 if(mode=='en'&&!r.has_en)mode='ne';
 $('txt').value=mode=='en'?cache[key].en_text:cache[key].text;
 $('dl').href=`/api/file/${jid}/${sel}`;$('dlEn').href=`/api/file/${jid}/${sel}?en=1`;
 $('dlEn').style.display=r.has_en?'':'none';$('tr').style.display=r.has_en?'none':'';
 $('tabEn').style.display=r.has_en?'':'none';
 renderTable(window.RS);
}
$('exgo').onclick=async()=>{
 $('log').textContent='';$('res').style.display='none';from=0;cache={};sel=0;
 const r=await fetch('/api/srt-job',{method:'POST',body:q({file:$('ex').value,clean:$('exclean').checked,en:$('exen').checked,en_model:$('enmodel').value})});
 const d=await r.json();if(!r.ok){$('log').innerHTML='<span class=err>'+d.error+'</span>';return}
 jid=d.id;$('go').disabled=true;clearInterval(timer);timer=setInterval(poll,1000);
};
async function loadSrts(){const l=await (await fetch('/api/srts')).json();
 $('ex').innerHTML=l.map(x=>`<option value="${x.name}">${x.name} - ${x.cues} cues, ${x.nepali_cues} with Nepali${x.has_en?' (has English)':''}</option>`).join('')||'<option>no SRTs yet</option>'}
loadSrts();
init();
</script>"""

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8765), H)
    print("UI at http://127.0.0.1:8765  (Ctrl+C to stop)  SRTs saved in ./outputs")
    threading.Timer(0.8, lambda: webbrowser.open("http://127.0.0.1:8765")).start()
    srv.serve_forever()
