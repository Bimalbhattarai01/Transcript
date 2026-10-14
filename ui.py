#!/usr/bin/env python3
"""Web UI + job queue: paste YouTube links (one, many, or a playlist/channel) -> jobs are queued in SQLite -> a worker downloads
the audio (yt-dlp), transcribes it (OpenRouter) and labels the speakers -> SRT. History lives in the database (data/app.db).

run:  python3 ui.py     then open http://127.0.0.1:8765
accounts: python3 manage.py adduser NAME  (login is required once a user exists, or when UI_REQUIRE_LOGIN=1 / UI_HOST is not loopback)
env:  UI_HOST, UI_PORT, UI_REQUIRE_LOGIN, DATA_DIR, SESSION_SECRET, YT_COOKIES (cookies.txt for yt-dlp), YT_PROXY,
      KEEP_AUDIO=0 (delete the mp3 after a job), DIARIZER, DIARIZE_SHIFT, DIARIZE_THREADS
"""
import hashlib, hmac, io, json, os, re, secrets, subprocess, sys, threading, time, traceback, webbrowser, zipfile
from argparse import Namespace
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import requests
import auth
import db
import transcribe as T  # also loads .env

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("DATA_DIR") or ROOT)
DL, OUT, DB_PATH = DATA / "downloads", DATA / "outputs", db.default_path()
REQUIRE_LOGIN = False  # set at startup: non-loopback host or UI_REQUIRE_LOGIN=1 (login is also on whenever a user exists)
DEFAULT_STT = ["qwen/qwen3-asr-1.7b"]
YT = re.compile(r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/")
_models = {"t": 0, "data": None}


# ---------- model list with prices (live from OpenRouter, no key needed) ----------
def stt_price(m):
    p = float(m["pricing"]["prompt"]); c = float(m["pricing"].get("completion") or 0)
    if c > 0: return f"token-based: ${p*1e6:.2f}/M in, ${c*1e6:.2f}/M out"
    if p < 0.01: return f"${p*3600:.2f}/hour"  # per-second pricing (matches vendor list prices for gpt-transcribe, chirp-3, nova-3)
    return f"${p:g} (unit unclear, check actual cost after run)"


def get_models():
    if _models["data"] and time.time() - _models["t"] < 600: return _models["data"]
    stt = []
    try:
        r = requests.get("https://openrouter.ai/api/v1/models?output_modalities=transcription", timeout=15).json()["data"]
        stt = [{"id": m["id"], "name": m["name"], "price": stt_price(m)} for m in r]
    except Exception as e:
        print("model list failed:", e)
    if not stt: stt = [{"id": i, "name": i, "price": "?"} for i in DEFAULT_STT]
    _models.update(t=time.time(), data={"stt": stt, "default_stt": DEFAULT_STT})
    return _models["data"]


# ---------- yt-dlp helpers ----------
def video_id(url):
    m = re.search(r"(?:youtu\.be/|[?&]v=|/shorts/|/live/)([\w-]{11})", url)
    return m.group(1) if m else None


def ytdlp_cmd():
    exe = next((str(p) for p in (DATA / "bin" / "yt-dlp", ROOT / "bin" / "yt-dlp") if p.exists()), "yt-dlp")
    cmd = [exe, "--js-runtimes", "node", "--remote-components", "ejs:github"]  # YouTube needs a JS runtime (node) + fresh yt-dlp
    if os.environ.get("YT_COOKIES"): cmd += ["--cookies", os.environ["YT_COOKIES"]]  # datacenter IPs are often asked to prove they're human
    if os.environ.get("YT_PROXY"): cmd += ["--proxy", os.environ["YT_PROXY"]]
    return cmd


def is_playlist(url):
    return "youtube.com/playlist" in url or bool(re.search(r"youtube\.com/(@[^/?]+|channel/[^/?]+|c/[^/?]+|user/[^/?]+)", url))


def expand_playlist(url):
    if re.search(r"youtube\.com/(@[^/?]+|channel/[^/?]+|c/[^/?]+|user/[^/?]+)/?$", url): url = url.rstrip("/") + "/videos"
    p = subprocess.run(ytdlp_cmd() + ["--flat-playlist", "--dump-single-json", url], capture_output=True, text=True, timeout=300)
    if p.returncode: raise RuntimeError(((p.stderr or "yt-dlp failed").strip().splitlines() or ["yt-dlp failed"])[-1][:200])

    def walk(e):
        for x in e.get("entries") or []:
            if x.get("entries"): yield from walk(x)
            elif x.get("id"): yield {"id": x["id"], "title": x.get("title"), "url": f"https://www.youtube.com/watch?v={x['id']}"}
    data = json.loads(p.stdout)
    title = re.sub(r"\s*[-–|]\s*(Videos|Shorts|Live|Playlists|Podcasts)\s*$", "", data.get("title") or "").strip()  # "Channel - Videos" -> "Channel"
    return title, list(walk(data))


def oembed_title(url):
    try:
        return requests.get("https://www.youtube.com/oembed", params={"url": url, "format": "json"}, timeout=8).json().get("title")
    except Exception:
        return None


# ---------- the worker ----------
class JobIO:
    """Writes a running job's log lines and progress into the database (progress writes are throttled)."""
    def __init__(self, jid): self.jid, self.t = jid, 0.0

    def log(self, msg): db.append_log(self.jid, str(msg))

    def set(self, force=False, **kw):
        if force or time.time() - self.t > 0.7:
            self.t = time.time(); db.update(self.jid, **kw)


def download(io, url, vid):
    if vid and (DL / f"{vid}.mp3").exists():  # same video again: skip the download
        io.log(f"Using cached audio {vid}.mp3")
        return str(DL / f"{vid}.mp3")
    io.log("Downloading audio with yt-dlp...")
    p = subprocess.Popen(ytdlp_cmd() + ["--no-playlist", "--newline", "-f", "bestaudio", "-x", "--audio-format", "mp3",
                                        "--print", "after_move:filepath", "-o", str(DL / "%(id)s.%(ext)s"), url],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    path = None
    for line in p.stdout:
        line = line.strip()
        if m := re.search(r"\[download\]\s+([\d.]+)%", line):
            io.set(progress=float(m[1]), detail=f"Downloading audio {float(m[1]):.0f}%")
        elif line.startswith("/") and line.endswith(".mp3"):
            path = line
        elif line and not line.startswith("[download]"):
            io.log(line)
    if p.wait() != 0 or not path: raise RuntimeError("yt-dlp failed (see log)")
    io.log(f"Downloaded {Path(path).name}")
    return path


def run_one(j):
    jid = j["id"]; io = JobIO(jid)
    db.update(jid, status="running", stage="download", progress=0, detail="Starting", started_at=time.time(), error=None)
    try:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key: raise RuntimeError("OPENROUTER_API_KEY missing in .env")
        if not j["title"]:
            t = oembed_title(j["url"])
            if t: db.update(jid, title=t)
        prior = db.find_done(j["video_id"], j["start_s"], j["end_s"], j["model"], j["speakers_hint"], j["label_speakers"], jid)
        if prior:
            io.log(f"Reused the finished result of job #{prior['id']} (same video, range and settings)")
            db.update(jid, status="done", stage="done", progress=100, detail="", srt_text=prior["srt_text"], cost=0, seconds=0,
                      cues=prior["cues"], words=prior["words"], finished_at=time.time())
            return
        DL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
        db.update(jid, detail="Downloading audio")
        path = download(io, j["url"], j["video_id"])
        io.set(True, stage="transcribe", progress=0, detail="Transcribing")
        T.hooks["log"] = io.log
        T.hooks["progress"] = lambda d, t: io.set(progress=100 * d / t, detail=f"Transcribing {d}/{t} chunks")
        limit = j["end_s"] - j["start_s"] if j["end_s"] else None
        tag = j["model"].replace("/", "_")
        a = Namespace(out=str(OUT), model=j["model"], lang="en", workers=6, max_chunk=20, soft_chunk=10, noise_db=-35, min_silence=0.35,
                      start=j["start_s"], limit_seconds=limit, compare=None, glossary=None, force=True, tag=tag,
                      speakers=j["speakers_hint"] or "" if j["label_speakers"] else None)
        c0, t0 = T.state["cost"], time.time()
        if not T.process(path, a, key, []): raise RuntimeError("transcription failed (see log)")
        srt = OUT / (Path(path).stem + f".{tag}" + (".test" if limit else "") + ".srt")
        text = srt.read_text(encoding="utf-8")
        db.update(jid, status="done", stage="done", progress=100, detail="", srt_text=text, cost=round(T.state["cost"] - c0, 4),
                  seconds=round(time.time() - t0), cues=text.count("-->"), words=len(text.split()), finished_at=time.time())
        if os.environ.get("KEEP_AUDIO", "1") == "0" and not db.q("SELECT 1 FROM jobs WHERE video_id=? AND status='queued'", (j["video_id"],)):
            Path(path).unlink(missing_ok=True)  # save disk on a server that processes hundreds of episodes
    except (Exception, SystemExit) as e:  # SystemExit: fatal API errors (bad key, no credits)
        io.log(f"ERROR: {e}")
        db.update(jid, status="error", stage="error", error=str(e)[:300], finished_at=time.time())
    finally:
        T.hooks["log"] = print; T.hooks["progress"] = lambda d, t: None


def worker():
    while True:
        try:
            j = db.next_queued()
            if not j: time.sleep(1.5); continue
            run_one(j)
        except Exception:
            traceback.print_exc(); time.sleep(3)


# ---------- auth (single shared password, signed cookie) ----------
def _secret():
    s = os.environ.get("SESSION_SECRET")
    if s: return s.encode()
    f = DB_PATH.parent / ".session_secret"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True); f.write_text(secrets.token_hex(32)); f.chmod(0o600)
    return f.read_text().strip().encode()


def _sig(user, exp, pw_hash):  # bound to the user's password hash: a password change logs that user out everywhere
    return hmac.new(_secret(), f"{user}|{exp}|{pw_hash[-24:]}".encode(), hashlib.sha256).hexdigest()


def make_cookie(user):
    exp = str(int(time.time()) + 30 * 86400)
    return f"{user['username']}.{exp}.{_sig(user['username'], exp, user['pw_hash'])}"


def user_from_cookie(v):
    """The (enabled) user a session cookie belongs to, or None. Re-read from the DB on every request."""
    try:
        name, exp, sig = v.rsplit(".", 2)
        u = db.get_user(name)
        if u and not u["disabled"] and int(exp) > time.time() and hmac.compare_digest(sig, _sig(u["username"], exp, u["pw_hash"])):
            return u
    except Exception:
        pass
    return None


def login_required():
    return REQUIRE_LOGIN or db.count_users() > 0


fails = {}  # ip -> recent failed login times


def project_arg(qs):
    """?project= : empty/absent = every job, 'none' = jobs without a project, a number = that project."""
    v = (qs.get("project") or [""])[0]
    return 0 if v == "none" else int(v) if v.isdigit() else None


def safe_name(s):
    return re.sub(r"[^\w\-. ()\[\]]+", "_", s or "transcript").strip(" ._")[:90] or "transcript"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json", extra=None):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def redirect(self, to, extra=None):
        self.send(302, "", "text/plain", {"Location": to, **(extra or {})})

    def ip(self):
        return (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()

    def authed(self):
        """True when the request may proceed; sets self.user (None when login is off, e.g. local development)."""
        self.user = None
        if not login_required(): return True
        c = SimpleCookie(self.headers.get("Cookie") or "")
        self.user = user_from_cookie(c["sess"].value) if "sess" in c else None
        return self.user is not None

    def cookie_header(self, value, max_age):
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
        return {"Set-Cookie": f"sess={value}; HttpOnly; SameSite=Lax; Path=/; Max-Age={max_age}{secure}"}

    def body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def jbody(self):
        try: return json.loads(self.body() or "{}")
        except ValueError: return {}

    def err(self, msg, code=400):
        return self.send(code, json.dumps({"error": msg}))

    # ---------- GET ----------
    def do_GET(self):
        u = urlparse(self.path); parts = u.path.strip("/").split("/"); qs = parse_qs(u.query)
        if u.path == "/static/Inter.woff2" and (ROOT / "static" / "Inter.woff2").exists():
            return self.send(200, (ROOT / "static" / "Inter.woff2").read_bytes(), "font/woff2", {"Cache-Control": "max-age=31536000"})
        if u.path == "/login": return self.send(200, LOGIN.replace("{{ERR}}", ""), "text/html; charset=utf-8") if login_required() else self.redirect("/")
        if u.path == "/logout": return self.redirect("/login", self.cookie_header("", 0))
        if not self.authed():
            return self.err("login required", 401) if u.path.startswith("/api/") else self.redirect("/login")
        if u.path == "/": return self.send(200, PAGE.replace("{{USER}}", self.user["username"] if self.user else ""), "text/html; charset=utf-8")
        if u.path == "/api/me": return self.send(200, json.dumps({"user": self.user["username"] if self.user else None}))
        if u.path == "/api/models": return self.send(200, json.dumps(get_models()))
        if u.path == "/api/jobs":
            return self.send(200, json.dumps({"jobs": db.list_jobs((qs.get("q") or [""])[0].strip(), project_arg(qs)), "counts": db.counts()}))
        if u.path == "/api/projects":
            return self.send(200, json.dumps({"projects": db.list_projects(), "unfiled": db.unfiled_count()}))
        if parts[:2] == ["api", "job"] and len(parts) == 3 and parts[2].isdigit():
            j = db.get(int(parts[2]), db.LIST_COLS + ", speakers_hint, log")
            if not j: return self.send(404, "{}")
            frm = int((qs.get("from") or ["0"])[0]); lines = j.pop("log").splitlines()
            return self.send(200, json.dumps({**j, "log": lines[frm:], "next": len(lines)}))
        if parts[:2] == ["api", "result"] and len(parts) == 3 and parts[2].isdigit():
            j = db.get(int(parts[2]), "srt_text")
            return self.send(200, json.dumps({"text": (j or {}).get("srt_text") or ""}))
        if parts[:2] == ["api", "file"] and len(parts) == 3 and parts[2].isdigit():
            j = db.get(int(parts[2]), "id, title, video_id, srt_text")
            if not j or not j["srt_text"]: return self.send(404, "{}")
            name = safe_name(j["title"] or j["video_id"]) + ".srt"
            return self.send(200, j["srt_text"], "application/x-subrip; charset=utf-8",
                             {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})
        if u.path == "/api/zip":
            ids = [int(i) for i in (qs.get("ids") or [""])[0].split(",") if i.isdigit()]
            rows = db.q("SELECT id, title, video_id, srt_text, project_id FROM jobs WHERE status='done' AND srt_text IS NOT NULL ORDER BY id")
            pr = project_arg(qs)
            rows = [r for r in rows if (not ids or r["id"] in ids) and (pr is None or (r["project_id"] or 0) == pr)]
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for r in rows: z.writestr(f"{safe_name(r['title'] or r['video_id'])} [{r['id']}].srt", r["srt_text"])
            return self.send(200, buf.getvalue(), "application/zip", {"Content-Disposition": 'attachment; filename="transcripts.zip"'})
        self.send(404, "{}")

    # ---------- POST ----------
    def do_POST(self):
        if self.path == "/login":
            ip, now = self.ip(), time.time()
            form = parse_qs(self.body().decode())
            name, pw = (form.get("username") or [""])[0].strip().lower(), (form.get("password") or [""])[0]
            keys = (ip, "user:" + name)
            for k in keys: fails[k] = [t for t in fails.get(k, []) if now - t < 300]
            if any(len(fails[k]) >= 5 for k in keys):
                return self.send(429, LOGIN.replace("{{ERR}}", "Too many attempts. Wait a few minutes."), "text/html; charset=utf-8")
            u = db.get_user(name) if name else None
            ok = auth.verify_password(pw, u["pw_hash"] if u else auth.DUMMY)  # always hash, so timing doesn't reveal valid usernames
            if u and ok and not u["disabled"]:
                db.touch_login(u["username"])
                return self.redirect("/", self.cookie_header(make_cookie(u), 30 * 86400))
            for k in keys: fails[k].append(now)
            return self.send(401, LOGIN.replace("{{ERR}}", "Wrong username or password."), "text/html; charset=utf-8")
        if not self.authed(): return self.err("login required", 401)
        d = self.jbody()
        if self.path == "/api/expand":
            items, skipped, expanded, seen, suggest = [], [], False, set(), ""
            for url in re.findall(r"https?://\S+", d.get("text") or ""):
                try:
                    if is_playlist(url) and not re.search(r"[?&]v=", url): ptitle, found = expand_playlist(url); expanded = True; suggest = suggest or ptitle
                    elif YT.match(url) and video_id(url): found = [{"id": video_id(url), "title": None, "url": url}]
                    else: skipped.append(url); continue
                except Exception as e:
                    return self.err(f"Could not read {url}: {e}")
                for it in found:
                    if it["id"] not in seen: seen.add(it["id"]); items.append(it)
            return self.send(200, json.dumps({"items": items, "expanded": expanded, "skipped": skipped, "suggest": suggest}))
        if self.path == "/api/enqueue":
            model = d.get("model") or ""
            if not re.fullmatch(r"[\w.\-]+/[\w.\-:]+", model): return self.err("Pick a transcription model")
            try:
                start = float(d.get("start") or 0)  # seconds
                end = float(d["end"]) if d.get("end") not in (None, "") else None
            except ValueError:
                return self.err("Start/end must be numbers")
            if end is not None and end <= start: return self.err("End time must be after start time")
            hint = ",".join(n.strip() for n in (d.get("speakers") or "").split(",") if n.strip())
            label = bool(d.get("label", True))
            items = [it for it in (d.get("items") or []) if YT.match(it.get("url") or "") and video_id(it["url"])]
            if not items: return self.err("Paste a valid YouTube link")
            project_id = None
            try:
                if (d.get("new_project") or "").strip(): project_id = db.add_project(d["new_project"], self.user["username"] if self.user else None)
                elif d.get("project_id"):
                    project_id = int(d["project_id"])
                    if not db.project_exists(project_id): return self.err("That project no longer exists")
            except ValueError as e:
                return self.err(str(e))
            batch = db.add_batch(f"{len(items)} videos") if len(items) > 1 else None
            ids, dup = [], 0
            for it in items:
                vid = video_id(it["url"])
                if db.active_duplicate(vid, start, end, model, hint, label): dup += 1; continue
                ids.append(db.add_job(batch, it["url"], vid, it.get("title"), start, end, model, hint, label,
                                      self.user["username"] if self.user else None, project_id))
            return self.send(200, json.dumps({"ids": ids, "queued": len(ids), "duplicates": dup, "project_id": project_id}))
        if self.path == "/api/retry":
            where = "status IN ('error','cancelled')" + ("" if d.get("all_failed") else " AND id=?")
            db.x(f"UPDATE jobs SET status='queued', stage='queued', progress=0, detail='', error=NULL, srt_text=NULL WHERE {where}",
                 () if d.get("all_failed") else (int(d.get("id") or 0),))
            return self.send(200, "{}")
        if self.path == "/api/cancel":
            where = "status='queued'" + ("" if d.get("all_queued") else " AND id=?")
            db.x(f"UPDATE jobs SET status='cancelled', stage='cancelled' WHERE {where}", () if d.get("all_queued") else (int(d.get("id") or 0),))
            return self.send(200, "{}")
        if self.path == "/api/delete":
            ids = d.get("ids") if isinstance(d.get("ids"), list) else [d.get("id") or 0]
            return self.send(200, json.dumps({"deleted": db.delete_jobs(ids)}))
        if self.path == "/api/move":
            pid = d.get("project_id")
            if pid is not None and not db.project_exists(int(pid)): return self.err("That project no longer exists")
            db.move_jobs(d.get("ids") or [], None if pid is None else int(pid))
            return self.send(200, "{}")
        if self.path == "/api/project":
            try:
                act = d.get("action")
                if act == "create": return self.send(200, json.dumps({"id": db.add_project(d.get("name"), self.user["username"] if self.user else None)}))
                if act == "rename": db.rename_project(int(d.get("id") or 0), d.get("name"))
                elif act == "delete": db.delete_project(int(d.get("id") or 0), bool(d.get("delete_jobs")))
                else: return self.err("unknown action")
            except ValueError as e:
                return self.err(str(e))
            return self.send(200, "{}")
        self.send(404, "{}")


LOGIN = r"""<!doctype html><meta charset=utf-8><title>Sign in &middot; Podcast SRT</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>@font-face{font-family:Inter;src:url(/static/Inter.woff2) format("woff2");font-weight:100 900;font-display:swap}
:root{--ink:#1b1f2e;--mut:#8a90a2;--sub:#6b7186;--line:#e4e7ee;--g:#1f7a5c}*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif;font-feature-settings:"cv11","ss03";-webkit-font-smoothing:antialiased;
 background:radial-gradient(900px 420px at 12% -10%,#e6e9fb,transparent 70%),radial-gradient(700px 360px at 100% 0,#eef0f8,transparent 70%),#f7f8fc}
.page{min-height:100vh;display:grid;grid-template-columns:1.1fr 1fr;align-items:center;gap:56px;max-width:1080px;margin:0 auto;padding:40px 28px}
.brand{font-weight:600;font-size:15px;position:absolute;top:30px;left:max(28px,calc((100vw - 1080px)/2 + 28px))}
.pitch h1{font-size:44px;line-height:1.1;font-weight:600;letter-spacing:-.03em;margin:0 0 16px;max-width:480px}
.pitch p{color:var(--mut);font-size:17px;margin:0;max-width:420px}
form{background:#fff;border:1px solid var(--line);border-radius:16px;padding:30px;box-shadow:0 1px 2px #1b1f2e08,0 18px 40px -22px #1b1f2e26;max-width:400px;width:100%;justify-self:end}
form h2{font-size:20px;font-weight:600;letter-spacing:-.02em;margin:0 0 4px}.hint{color:var(--mut);font-size:13.5px;margin:0 0 22px}
label{display:block;font-size:13px;font-weight:500;color:var(--sub);margin:0 0 6px}
.field{position:relative;margin-bottom:16px}
input{font:inherit;font-size:14.5px;width:100%;height:44px;padding:0 12px;border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--ink);outline:none;transition:border-color .12s,box-shadow .12s}
input::placeholder{color:#b0b5c3}input:focus{border-color:var(--g);box-shadow:0 0 0 3px #1f7a5c1a}
.pw{position:relative}.pw input{padding-right:46px}
.eye{position:absolute;right:5px;top:5px;width:34px;height:34px;padding:0;border:0;background:none;color:#8a90a2;cursor:pointer;border-radius:8px;display:grid;place-items:center}.eye:hover{background:#f1f2f8;color:var(--ink)}.eye svg{width:18px;height:18px;display:block}.eye .off{display:none}.eye.on .off{display:block}.eye.on .see{display:none}
.go{font:inherit;font-size:14.5px;font-weight:500;width:100%;height:46px;border:0;border-radius:10px;background:var(--g);color:#fff;cursor:pointer;margin-top:6px;transition:background .12s}
.go:hover{background:#196a4f}.go:disabled{opacity:.6;cursor:default}
.e{background:#fdf0ef;color:#b3302a;font-size:13px;border-radius:9px;padding:9px 12px;margin:0 0 16px}.e:empty{display:none}
.caps{color:#b07a12;font-size:12.5px;margin:-8px 0 12px;display:none}
.foot{color:var(--mut);font-size:13px;margin:18px 0 0;text-align:center}
@media(max-width:860px){.page{grid-template-columns:1fr;gap:28px;padding:92px 20px 40px}.brand{left:20px}.pitch h1{font-size:32px}.pitch p{font-size:15.5px}form{justify-self:stretch;max-width:none}}</style>
<div class=brand>Podcast SRT</div>
<div class=page>
 <div class=pitch><h1>Turn a YouTube link into a speaker-labelled SRT</h1><p>Paste a link, or a whole playlist, and get subtitles that say who is speaking. For the team only.</p></div>
 <form method=post action=/login id=f>
  <h2>Sign in</h2><p class=hint>Use the email or username your admin gave you.</p>
  <p class=e role=alert>{{ERR}}</p>
  <div class=field><label for=u>Email or username</label><input id=u name=username autocomplete=username autocapitalize=off spellcheck=false autofocus required></div>
  <div class=field><label for=p>Password</label><div class=pw><input id=p type=password name=password autocomplete=current-password required>
   <button type=button class=eye id=eye aria-label="Show password" title="Show password">
    <svg class=see viewBox="0 0 24 24" fill=none stroke=currentColor stroke-width=1.7 stroke-linecap=round stroke-linejoin=round><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx=12 cy=12 r=3 /></svg>
    <svg class=off viewBox="0 0 24 24" fill=none stroke=currentColor stroke-width=1.7 stroke-linecap=round stroke-linejoin=round><path d="M17.9 17.9A10.4 10.4 0 0 1 12 19c-6.4 0-10-7-10-7a18.3 18.3 0 0 1 4.1-5M9.9 5.2A9.7 9.7 0 0 1 12 5c6.4 0 10 7 10 7a18.3 18.3 0 0 1-2.2 3.2M1 1l22 22M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>
   </button></div></div>
  <div class=caps id=caps>Caps Lock is on</div>
  <button class=go id=go>Sign in</button>
  <p class=foot>No account? Ask the person who runs this server to add you.</p>
 </form>
</div>
<script>
const p=document.getElementById('p'),eye=document.getElementById('eye');
eye.onclick=()=>{const show=p.type=='password';p.type=show?'text':'password';eye.classList.toggle('on',show);const t=show?'Hide password':'Show password';eye.setAttribute('aria-label',t);eye.title=t;p.focus()};
p.addEventListener('keyup',e=>{document.getElementById('caps').style.display=e.getModifierState&&e.getModifierState('CapsLock')?'block':'none'});
document.getElementById('f').addEventListener('submit',()=>{const b=document.getElementById('go');b.disabled=true;b.textContent='Signing in...'});
</script>"""


PAGE = r"""<!doctype html><meta charset=utf-8><title>Podcast SRT</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
@font-face{font-family:Inter;src:url(/static/Inter.woff2) format("woff2");font-weight:100 900;font-display:swap}
:root{--ink:#1b1f2e;--mut:#8a90a2;--sub:#6b7186;--line:#e4e7ee;--g:#1f7a5c;--red:#c8372d}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif;font-feature-settings:"cv11","ss03";-webkit-font-smoothing:antialiased;
 background:radial-gradient(900px 380px at 25% -8%,#e8ebfb,transparent 70%),#f7f8fc}
input,select,button,textarea{font-family:Inter,system-ui,sans-serif}
svg{width:16px;height:16px;display:block;flex:none}
.app{display:grid;grid-template-columns:300px 1fr;min-height:100vh}
aside{border-right:1px solid var(--line);background:#fbfbfe;padding:22px 14px 16px;display:flex;flex-direction:column;gap:10px;position:sticky;top:0;height:100vh}
.brand{font-weight:600;font-size:15px;padding:0 8px 6px}
aside .ghost{width:100%;height:38px;background:#fff;color:var(--ink);border:1px solid var(--line);font-size:13.5px}aside .ghost:hover{background:#f3f4f9}
#search{height:36px;font-size:13px}
.shead{display:flex;align-items:center;justify-content:space-between;font-size:11.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);padding:8px 8px 2px}
.shead button{height:24px;width:24px;padding:0;background:none;color:var(--sub);border-radius:7px;display:grid;place-items:center;font-size:17px;line-height:1;font-weight:400}.shead button:hover{background:#eceef7;color:var(--ink)}
.plist{max-height:208px;overflow-y:auto;overflow-x:hidden;margin:0 -6px;padding:0 6px}
.prow{display:grid;grid-template-columns:16px 1fr auto auto;gap:9px;align-items:center;padding:7px 8px;border-radius:9px;cursor:pointer;font-size:13.5px;color:var(--ink)}
.prow:hover{background:#f1f2f8}.prow.on{background:#eceef7;font-weight:500}.prow svg{color:var(--sub)}
.prow span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.prow em{font-style:normal;font-size:12px;color:var(--mut)}
.ib{height:24px;width:24px;padding:0;border-radius:7px;background:none;color:var(--sub);display:none;place-items:center;border:0;cursor:pointer}.ib:hover{background:#e3e5f0;color:var(--ink)}.ib.del:hover{color:var(--red)}
.prow .acts{display:none;gap:0}.prow:hover .acts{display:flex}.prow:hover em{display:none}.acts .ib{display:grid}
.list{flex:1;overflow-y:auto;overflow-x:hidden;margin:0 -6px;padding:0 6px;min-height:120px}
.item{display:grid;grid-template-columns:8px 1fr 26px;gap:9px;padding:9px 6px 9px 8px;border-radius:9px;cursor:pointer;align-items:start}.item:hover{background:#f1f2f8}.item.on{background:#eceef7}
.item .ib{display:none}.item:hover .ib{display:grid}
.dot{width:8px;height:8px;border-radius:50%;margin-top:6px;background:#c8ccd8}.dot.running{background:#e8a33b;box-shadow:0 0 0 3px #e8a33b33}.dot.done{background:var(--g)}.dot.error{background:#d6453d}.dot.queued{background:#9aa1b5}
.item b{font-weight:500;font-size:13.5px;display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.item small{color:var(--mut);font-size:12px;display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.nolist{color:var(--mut);font-size:13px;padding:16px 8px}
.sfoot{border-top:1px solid var(--line);padding-top:12px;font-size:12.5px;color:var(--mut);display:flex;flex-direction:column;gap:8px}
.sfoot .r{display:flex;gap:12px;flex-wrap:wrap}.sfoot a,.sfoot button.lnk{color:var(--sub);background:none;border:0;padding:0;height:auto;font-size:12.5px;cursor:pointer;text-decoration:underline}
main{padding:0 44px 70px;width:100%;min-width:0;max-width:1600px}
.topbar{display:flex;justify-content:flex-end;align-items:center;height:64px;position:relative}
.menu{position:relative}
.user{height:38px;padding:0 10px 0 5px;gap:9px;display:flex;align-items:center;background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:999px;font-size:13.5px;font-weight:500}.user:hover{background:#f5f6fa}
.av{width:28px;height:28px;border-radius:50%;background:var(--g);color:#fff;display:grid;place-items:center;font-size:12.5px;font-weight:600;text-transform:uppercase}
.user svg{color:var(--mut);width:14px;height:14px}
.drop{position:absolute;right:0;top:46px;min-width:250px;background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 14px 34px -14px #1b1f2e40;padding:6px;z-index:30}
.drop .who2{padding:10px 12px 12px;border-bottom:1px solid #eef0f5;margin-bottom:4px}.drop small{display:block;color:var(--mut);font-size:12px}.drop b{font-weight:500;font-size:13.5px;overflow-wrap:anywhere}
.drop a{display:flex;align-items:center;gap:9px;padding:9px 12px;border-radius:8px;color:var(--ink);text-decoration:none;font-size:13.5px}.drop a:hover{background:#f3f4f9}.drop a svg{color:var(--sub)}
h1{font-size:34px;line-height:1.18;font-weight:600;letter-spacing:-.025em;margin:6px 0 12px;max-width:640px}
.lede{color:var(--mut);font-size:16px;margin:0 0 28px;max-width:560px}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:24px;box-shadow:0 1px 2px #1b1f2e08}
label{display:block;font-size:13px;font-weight:500;color:var(--sub);margin:0 0 6px}label i{font-style:normal;color:var(--mut);font-weight:400}
input,select,textarea{width:100%;height:42px;font-size:14px;color:var(--ink);padding:0 12px;border:1px solid var(--line);border-radius:10px;background:#fff;outline:none;transition:border-color .12s,box-shadow .12s}
textarea{height:112px;padding:10px 12px;resize:vertical;line-height:1.5}
input::placeholder,textarea::placeholder{color:#a9aebd}input:focus,select:focus,textarea:focus{border-color:var(--g);box-shadow:0 0 0 3px #1f7a5c1a}
.f{margin-bottom:16px}.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.seg{display:inline-flex;background:#f1f2f8;border-radius:10px;padding:3px;margin-bottom:18px}.seg button{height:30px;padding:0 14px;border-radius:8px;background:none;color:var(--sub);font-size:13px}.seg button.on{background:#fff;color:var(--ink);box-shadow:0 1px 2px #1b1f2e14}
.chk{display:flex;align-items:center;gap:8px;font-size:13.5px;color:var(--sub);margin:2px 0 0;font-weight:400}.chk input{width:16px;height:16px;accent-color:var(--g);padding:0}
.act{display:flex;align-items:center;gap:16px;margin-top:20px;flex-wrap:wrap}
button{font-size:14px;font-weight:500;height:42px;padding:0 20px;border:0;border-radius:10px;background:var(--g);color:#fff;cursor:pointer;white-space:nowrap;flex:none}button:hover{background:#196a4f}button:disabled{opacity:.5;cursor:default}
button.quiet{background:#fff;color:var(--ink);border:1px solid var(--line);height:36px;padding:0 14px;font-size:13px}button.quiet:hover{background:#f5f6fa}button.danger{color:var(--red)}
button.dangerbtn{background:var(--red);height:36px;padding:0 14px;font-size:13px}button.dangerbtn:hover{background:#a82c24}
select.mini{height:36px;width:auto;max-width:200px;font-size:13px;padding:0 10px}
.hint{color:var(--mut);font-size:13px}.err{color:var(--red);font-size:13px}.ok{color:var(--g);font-size:13px}
#review{margin-top:18px}.rv{max-height:300px;overflow:auto;margin:10px -8px 0;padding:0 8px}.rv label{display:flex;gap:10px;align-items:center;padding:8px 0;border-top:1px solid #eef0f5;margin:0;font-weight:400;color:var(--ink);font-size:13.5px}.rv input{width:16px;height:16px;accent-color:var(--g);flex:none}
.top{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin:6px 0 4px}.top h2{font-size:24px;font-weight:600;letter-spacing:-.02em;margin:0;line-height:1.25;min-width:0;overflow-wrap:anywhere}
.tools{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.meta{color:var(--mut);font-size:13px;margin:6px 0 18px}.meta a{color:var(--sub)}
.bar{height:4px;border-radius:4px;background:#e8eaf1;overflow:hidden}.bar i{display:block;height:100%;width:0;background:var(--g);transition:width .4s}
.status{display:flex;justify-content:space-between;margin-top:10px;font-size:13px;color:var(--sub)}.status.bad{color:var(--red)}
details{margin-top:12px;font-size:13px;color:var(--mut)}summary{cursor:pointer}
pre{margin:8px 0 0;padding:12px;background:#f5f6fa;border-radius:10px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--sub);max-height:200px;overflow-wrap:anywhere;overflow:auto;white-space:pre-wrap}
.stats{display:flex;gap:34px;margin:22px 0 18px}.stats div{font-size:12.5px;color:var(--mut)}
.stats b{display:block;font-size:20px;font-weight:600;color:var(--ink);letter-spacing:-.01em;font-variant-numeric:tabular-nums;margin-top:2px}.stats small{font-size:13px;font-weight:500;color:var(--mut)}
.hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}.hd h3{margin:0;font-size:15px;font-weight:600}
a.dl{font-size:13px;font-weight:500;color:var(--ink);text-decoration:none;border:1px solid var(--line);border-radius:9px;padding:7px 12px}a.dl:hover{background:#f5f6fa}
.rows{max-height:68vh;overflow:auto;margin:0 -8px;padding:0 8px}
.row{display:grid;grid-template-columns:48px 168px 1fr;gap:12px;padding:12px 0;border-top:1px solid #eef0f5;font-size:14px;align-items:baseline}.row:first-child{border-top:0}
.t{color:var(--mut);font-size:12.5px;font-variant-numeric:tabular-nums}.who{font-size:13px;font-weight:500;color:var(--sub);display:flex;align-items:center;gap:7px}.who b{width:6px;height:6px;border-radius:50%;flex:none}
.x{white-space:pre-line}.empty{color:var(--mut);padding:26px 0;text-align:center;font-size:14px}
dialog{border:1px solid var(--line);border-radius:16px;padding:24px;width:min(430px,92vw);color:var(--ink);box-shadow:0 30px 60px -20px #1b1f2e55}dialog::backdrop{background:#1b1f2e59}
dialog h3{margin:0 0 6px;font-size:17px;font-weight:600;letter-spacing:-.01em}dialog p{margin:0 0 14px;color:var(--sub);font-size:14px;overflow-wrap:anywhere}dialog input{margin-top:6px}
.dact{display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap;margin-top:18px}
@media(max-width:820px){.app{grid-template-columns:1fr}aside{position:static;height:auto;max-height:440px}main{padding:0 18px 60px}.two{grid-template-columns:1fr}.row{grid-template-columns:40px 1fr}.row .x{grid-column:1/-1}}
</style>
<div class=app>
<aside>
 <div class=brand>Podcast SRT</div>
 <button class=ghost id=newbtn>+ New transcript</button>
 <input id=search placeholder="Search history">
 <div class=shead><span>Projects</span><button id=newproj_btn title="New project" aria-label="New project">+</button></div>
 <div class=plist id=plist></div>
 <div class=shead><span id=lhead>All transcripts</span></div>
 <div class=list id=list><div class=nolist>No transcripts yet.</div></div>
 <div class=sfoot><span id=qsum>&nbsp;</span>
  <div class=r><a href=/api/zip id=zip>Download all (.zip)</a><button class=lnk id=retryall hidden>Retry failed</button></div></div>
</aside>
<main>
 <div class=topbar>
  <div class=menu id=umenu hidden>
   <button class=user id=ubtn aria-haspopup=menu aria-expanded=false><span class=av id=av></span><span id=uname></span>
    <svg viewBox="0 0 24 24" fill=none stroke=currentColor stroke-width=2 stroke-linecap=round stroke-linejoin=round><path d="m6 9 6 6 6-6"/></svg></button>
   <div class=drop id=udrop hidden role=menu>
    <div class=who2><small>Signed in as</small><b id=uemail></b></div>
    <a href=/logout role=menuitem><svg viewBox="0 0 24 24" fill=none stroke=currentColor stroke-width=1.8 stroke-linecap=round stroke-linejoin=round><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/></svg>Log out</a>
   </div>
  </div>
 </div>
<section id=pNew>
 <h1>Turn a YouTube link into a speaker-labelled SRT</h1>
 <p class=lede>Paste one link, many links, or a whole playlist or channel. Every subtitle line comes back with who is speaking.</p>
 <div class=card>
  <div class=seg><button class=on id=m1>One video</button><button id=mN>Many videos</button></div>
  <div class=f id=f1><label>YouTube link</label><input id=url placeholder="https://www.youtube.com/watch?v=..."></div>
  <div class=f id=fN hidden><label>Links <i>(one per line; playlist and channel links are expanded)</i></label><textarea id=urls placeholder="https://www.youtube.com/watch?v=...&#10;https://www.youtube.com/playlist?list=..."></textarea></div>
  <div class="two f">
   <div><label>Start time</label><select id=start></select></div>
   <div><label>End time</label><select id=end></select></div>
  </div>
  <div class=hint style="margin:-4px 0 16px">Tip: choose an End time like 0:30 for a quick test.</div>
  <div class="two f">
   <div><label>Model</label><select id=model><option>Loading models...</option></select></div>
   <div><label>Speaker names <i>(optional)</i></label><input id=spknames placeholder="Host Name, Guest Name"></div>
  </div>
  <div class="two f">
   <div><label>Project <i>(e.g. one per channel)</i></label><select id=proj></select></div>
   <div id=npw hidden><label>New project name</label><input id=newproj placeholder="Channel or show name" maxlength=60></div>
  </div>
  <label class=chk><input type=checkbox id=label checked> Label speakers <span class=hint>(slower; turn off for plain text)</span></label>
  <div class=act><button id=go>Generate SRT</button><span class=err id=err></span><span class=ok id=okmsg></span></div>
 </div>
 <div class=card id=review hidden>
  <div class=hd><h3 id=rvtitle></h3><button class=quiet id=rvall>Untick all</button></div>
  <div class=rv id=rvlist></div>
  <div class=act><button id=rvgo>Queue selected</button><span class=err id=rverr></span></div>
 </div>
</section>

<section id=pJob hidden>
 <div class=top><h2 id=jtitle>&nbsp;</h2>
  <div class=tools><select class=mini id=jproj title="Move to project"></select><button class=quiet id=jretry hidden>Retry</button><button class=quiet id=jcancel hidden>Cancel</button><button class="quiet danger" id=jdel>Delete</button></div></div>
 <div class=meta id=jmeta></div>
 <div id=jrun>
  <div class=bar><i id=bar></i></div>
  <div class=status id=jstatus><span id=detail>Queued</span><span id=stage></span></div>
 </div>
 <details><summary>Log</summary><pre id=log></pre></details>
 <div id=jout hidden>
  <div class=stats>
   <div>Cost<b id=s_cost>&ndash;</b></div><div>Time<b id=s_time>&ndash;</b></div><div>Subtitle lines<b id=s_cues>&ndash;</b></div><div>Words<b id=s_words>&ndash;</b></div>
  </div>
  <div class=card>
   <div class=hd><h3>Transcript</h3><a class=dl id=dl href=#>Download .srt</a></div>
   <div class=rows id=rows></div>
  </div>
 </div>
</section>
</main>
</div>
<dialog id=dlg><h3 id=dtitle></h3><p id=dtext></p><input id=dinput hidden><div class=dact id=dbtns></div></dialog>
<script>
const $=id=>document.getElementById(id);
const COL=['#4f5bd5','#e8743b','#1f7a5c','#2d9cc4','#c79a1f','#b8374a'],esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const S='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">';
const IC={folder:S+'<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>',
 all:S+'<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18"/></svg>',
 edit:S+'<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
 trash:S+'<path d="M3 6h18M8 6V4h8v2m-9 0 1 14h8l1-14M10 11v6M14 11v6"/></svg>'};
let sel=null,mode='one',logFrom=0,loaded=null,lastJob=null,review=[],suggest='',cur='all',projects=[],unfiled=0,jobs=[],counts={};
const ME="{{USER}}";
const fmt=t=>{t=Math.round(t);const h=Math.floor(t/3600),m=Math.floor(t%3600/60),x=t%60;return (h?h+':'+String(m).padStart(2,'0'):m)+':'+String(x).padStart(2,'0')};
const ago=t=>{const s=Date.now()/1000-t;return s<60?'just now':s<3600?Math.floor(s/60)+' min ago':s<86400?Math.floor(s/3600)+' h ago':Math.floor(s/86400)+' d ago'};
const api=async(p,body)=>{const r=await fetch(p,body===undefined?{}:{method:'POST',body:JSON.stringify(body)});if(r.status==401){location='/login';throw 0}
 let d={};try{d=await r.json()}catch(e){}return {ok:r.ok,d}};

/* ---- small modal (replaces confirm/prompt) ---- */
function modal({title,text='',input=null,buttons}){return new Promise(res=>{
 $('dtitle').textContent=title;$('dtext').textContent=text;$('dtext').hidden=!text;
 const inp=$('dinput');inp.hidden=!input;if(input){inp.value=input.value||'';inp.placeholder=input.placeholder||''}
 $('dbtns').innerHTML=buttons.map((b,i)=>`<button type=button class="${b.kind||'quiet'}" data-i="${i}">${esc(b.label)}</button>`).join('');
 const d=$('dlg'),done=v=>{d.close();res(v)};
 $('dbtns').querySelectorAll('button').forEach(b=>b.onclick=()=>done({v:buttons[+b.dataset.i].value,text:inp.value.trim()}));
 d.oncancel=()=>res({v:null,text:''});
 inp.onkeydown=e=>{if(e.key=='Enter'){e.preventDefault();const p=buttons.find(b=>b.primary);if(p)done({v:p.value,text:inp.value.trim()})}};
 d.showModal();if(input)inp.focus()})}
const info=(title,text)=>modal({title,text,buttons:[{label:'OK',kind:'',value:1,primary:true}]});

/* ---- user menu ---- */
function userUI(){if(!ME)return;$('umenu').hidden=false;$('av').textContent=ME[0];$('uname').textContent=ME.includes('@')?ME.split('@')[0]:ME;$('uemail').textContent=ME}
const closeMenu=()=>{$('udrop').hidden=true;$('ubtn').setAttribute('aria-expanded','false')};
$('ubtn').onclick=e=>{e.stopPropagation();const open=$('udrop').hidden;$('udrop').hidden=!open;$('ubtn').setAttribute('aria-expanded',open)};
document.addEventListener('click',e=>{if(!$('umenu').contains(e.target))closeMenu()});document.addEventListener('keydown',e=>{if(e.key=='Escape')closeMenu()});

function timeUI(){ // 30 s steps for the first 10 min (short tests), then every minute up to 4 h
 const ts=[];for(let t=0;t<=14400;t+=t<600?30:60)ts.push(t);
 const o=ts.map(t=>`<option value="${t}">${fmt(t)}</option>`).join('');
 $('start').innerHTML=o;$('end').innerHTML='<option value="">End of video</option>'+o;
}
async function init(){
 timeUI();userUI();
 const {d:M}=await api('/api/models');
 $('model').innerHTML=M.stt.map(m=>`<option value="${m.id}">${esc(m.name)} · ${esc(m.price)}</option>`).join('');
 if(M.stt.some(m=>m.id==M.default_stt[0]))$('model').value=M.default_stt[0];
 await loadProjects();await loadList();setInterval(tick,2500);
}
function setMode(m){mode=m;$('m1').className=m=='one'?'on':'';$('mN').className=m=='many'?'on':'';$('f1').hidden=m!='one';$('fN').hidden=m!='many';
 $('go').textContent=m=='one'?'Generate SRT':'Add to queue';$('review').hidden=true;$('err').textContent=$('okmsg').textContent=''}
$('m1').onclick=()=>setMode('one');$('mN').onclick=()=>setMode('many');
function showNew(){sel=null;loaded=null;$('pNew').hidden=false;$('pJob').hidden=true;renderList()}
$('newbtn').onclick=showNew;

/* ---- projects ---- */
async function loadProjects(){
 const {d}=await api('/api/projects');projects=d.projects;unfiled=d.unfiled;
 if(typeof cur=='number'&&!projects.some(p=>p.id==cur))cur='all';
 renderProjects();fillSelects()}
function renderProjects(){
 const total=Object.values(counts).reduce((a,b)=>a+b,0);
 const row=(key,icon,name,n,acts)=>`<div class="prow ${String(cur)==String(key)?'on':''}" data-p="${key}">${icon}<span title="${esc(name)}">${esc(name)}</span><em>${n??''}</em><span class=acts>${acts||''}</span></div>`;
 $('plist').innerHTML=row('all',IC.all,'All transcripts',total||'')+projects.map(p=>row(p.id,IC.folder,p.name,p.total,
  `<button class=ib data-act=rename data-id="${p.id}" title="Rename" aria-label="Rename">${IC.edit}</button><button class="ib del" data-act=delete data-id="${p.id}" title="Delete project" aria-label="Delete project">${IC.trash}</button>`)).join('')
  +(unfiled?row('none',IC.folder,'No project',unfiled):'');
 $('plist').querySelectorAll('.prow').forEach(e=>e.onclick=ev=>{
  const b=ev.target.closest('[data-act]');if(b){ev.stopPropagation();return projectAction(b.dataset.act,+b.dataset.id)}
  const k=e.dataset.p;cur=k=='all'||k=='none'?k:+k;pickProject()});
 $('zip').href='/api/zip'+(cur=='all'?'':'?project='+cur);
 $('lhead').textContent=cur=='all'?'All transcripts':cur=='none'?'No project':(projects.find(p=>p.id==cur)||{}).name||'Transcripts';
}
function fillSelects(){
 const opts=(extra)=>'<option value="">No project</option>'+projects.map(p=>`<option value="${p.id}">${esc(p.name)}</option>`).join('')+(extra||'');
 const keepF=$('proj').value,keepJ=$('jproj').value;
 $('proj').innerHTML=opts('<option value="__new">+ New project…</option>');$('jproj').innerHTML=opts();
 $('proj').value=[...$('proj').options].some(o=>o.value==keepF)?keepF:(typeof cur=='number'?String(cur):'');
 if(document.activeElement!==$('jproj')&&lastJob)$('jproj').value=lastJob.project_id==null?'':String(lastJob.project_id);else $('jproj').value=keepJ;
 $('npw').hidden=$('proj').value!='__new';
}
function pickProject(){ // clicking a project filters the list and makes it the default for new transcripts
 renderProjects();loadList();if(typeof cur=='number')$('proj').value=String(cur);else if($('proj').value!='__new')$('proj').value='';$('npw').hidden=$('proj').value!='__new'}
$('proj').onchange=()=>{$('npw').hidden=$('proj').value!='__new';if(!$('npw').hidden)$('newproj').focus()};
$('newproj_btn').onclick=async()=>{
 const r=await modal({title:'New project',text:'Group transcripts, for example one project per channel or show.',input:{placeholder:'Project name'},buttons:[{label:'Cancel',value:'no'},{label:'Create',kind:'',value:'ok',primary:true}]});
 if(r.v!='ok'||!r.text)return;const {ok,d}=await api('/api/project',{action:'create',name:r.text});
 if(!ok)return info('Could not create the project',d.error);await loadProjects();cur=d.id;pickProject()};
async function projectAction(act,id){
 const p=projects.find(x=>x.id==id);if(!p)return;
 if(act=='rename'){const r=await modal({title:'Rename project',input:{value:p.name},buttons:[{label:'Cancel',value:'no'},{label:'Save',kind:'',value:'ok',primary:true}]});
  if(r.v!='ok'||!r.text||r.text==p.name)return;const {ok,d}=await api('/api/project',{action:'rename',id,name:r.text});if(!ok)return info('Could not rename',d.error);await loadProjects();return}
 const n=p.total,bs=[{label:'Cancel',value:'no'},{label:n?'Delete project only':'Delete project',kind:n?'quiet':'dangerbtn',value:'keep',primary:!n}];
 if(n)bs.push({label:`Delete project and ${n} transcript${n==1?'':'s'}`,kind:'dangerbtn',value:'all'});
 const r=await modal({title:`Delete “${p.name}”?`,text:n?`It has ${n} transcript${n==1?'':'s'}. “Delete project only” keeps them under No project.`:'The project is empty.',buttons:bs});
 if(!r.v||r.v=='no')return;await api('/api/project',{action:'delete',id,delete_jobs:r.v=='all'});
 if(cur==id)cur='all';if(sel!=null&&r.v=='all'&&lastJob&&lastJob.project_id==id)showNew();await refresh()}

/* ---- history list ---- */
async function loadList(){
 const {d}=await api('/api/jobs?q='+encodeURIComponent($('search').value.trim())+(cur=='all'?'':'&project='+cur));jobs=d.jobs;counts=d.counts;renderList();
 const c=counts;$('qsum').textContent=[c.running?'1 running':'',c.queued?c.queued+' queued':'',c.done+' done',c.error?c.error+' failed':''].filter(Boolean).join(' · ');
 $('retryall').hidden=!c.error;
}
const refresh=async()=>{await loadProjects();await loadList()};
function sub(j){const pj=cur=='all'&&j.project_id?(projects.find(p=>p.id==j.project_id)||{}).name:'';
 const by=[pj,j.created_by].filter(Boolean).join(' · ');
 return [j.status=='running'?(j.detail||'Running'):j.status=='done'?`${j.cues} lines · ${ago(j.finished_at||j.created_at)}`:j.status=='error'?'Failed':j.status=='queued'?'Queued':'Cancelled',by].filter(Boolean).join(' · ')}
function renderList(){
 $('list').innerHTML=jobs.length?jobs.map(j=>`<div class="item ${sel==j.id?'on':''}" data-id="${j.id}"><span class="dot ${j.status}"></span><span><b>${esc(j.title||j.video_id||j.url)}</b><small>${esc(sub(j))}</small></span>${j.status=='running'?'<span></span>':`<button class="ib del" data-del="${j.id}" title="Delete" aria-label="Delete">${IC.trash}</button>`}</div>`).join(''):'<div class=nolist>No transcripts here yet.</div>';
 $('list').querySelectorAll('.item').forEach(e=>e.onclick=ev=>{const b=ev.target.closest('[data-del]');if(b){ev.stopPropagation();const j=jobs.find(x=>x.id==+b.dataset.del);return delJob(j.id,j.title||j.video_id)}openJob(+e.dataset.id)});
 renderProjects();
}
let st;$('search').oninput=()=>{clearTimeout(st);st=setTimeout(loadList,250)};
async function tick(){await loadProjects();await loadList();if(sel!=null&&(!lastJob||['queued','running'].includes(lastJob.status)))await pollJob()}
async function delJob(id,title){
 const r=await modal({title:'Delete this transcript?',text:title||'',buttons:[{label:'Cancel',value:'no'},{label:'Delete',kind:'dangerbtn',value:'yes',primary:true}]});
 if(r.v!='yes')return;await api('/api/delete',{ids:[id]});if(sel==id)showNew();await refresh()}

/* ---- job view ---- */
async function openJob(id){sel=id;logFrom=0;loaded=null;lastJob=null;$('log').textContent='';$('pNew').hidden=true;$('pJob').hidden=false;$('jout').hidden=true;$('rows').innerHTML='';renderList();await pollJob()}
async function pollJob(){
 const id=sel,{ok,d:j}=await api(`/api/job/${id}?from=${logFrom}`);if(!ok||id!=sel)return;lastJob=j;logFrom=j.next;
 if(j.log.length){$('log').textContent+=j.log.join('\n')+'\n';$('log').scrollTop=1e9}
 $('jtitle').textContent=j.title||j.video_id||j.url;
 if(document.activeElement!==$('jproj'))$('jproj').value=j.project_id==null?'':String(j.project_id);
 const range=(j.start_s||j.end_s)?` · ${fmt(j.start_s||0)}–${j.end_s?fmt(j.end_s):'end'}`:'';
 $('jmeta').innerHTML=`<a href="${esc(j.url)}" target=_blank rel=noopener>Open on YouTube</a> · ${esc(j.model)}${range}${j.label_speakers?'':' · no speaker labels'} · added ${ago(j.created_at)}${j.created_by?' by '+esc(j.created_by):''}`;
 const live=j.status=='queued'||j.status=='running';
 $('jrun').hidden=j.status=='done';
 $('bar').style.width=(j.status=='error'?0:j.progress||0)+'%';
 $('detail').textContent=j.status=='error'?(j.error||'Failed. See the log.'):j.status=='cancelled'?'Cancelled':j.status=='queued'?'Waiting in the queue':(j.detail||'Working...');
 $('jstatus').className='status'+(j.status=='error'?' bad':'');
 $('stage').textContent=live?j.stage:'';
 $('jretry').hidden=!(j.status=='error'||j.status=='cancelled');$('jcancel').hidden=j.status!='queued';$('jdel').hidden=j.status=='running';
 if(j.status=='done'){
  $('s_cost').textContent='$'+(j.cost??0);$('s_time').innerHTML=(j.seconds??0)+' <small>s</small>';$('s_cues').textContent=j.cues;$('s_words').textContent=j.words;
  if(loaded!=id){loaded=id;const {d}=await api(`/api/result/${id}`);if(id==sel)showRows(d.text,id)}
 }
}
function showRows(text,id){
 $('dl').href=`/api/file/${id}`;$('jout').hidden=false;const who=[];
 $('rows').innerHTML=text.trim().split(/\n\n+/).map(b=>{const l=b.split('\n'),t=(l[1]||'').slice(3,8),body=l.slice(2).join('\n'),m=body.match(/^([^:\n]{2,40}):\s+([\s\S]*)/);
  const sp=m?m[1]:'',tx=m?m[2]:body;if(sp&&!who.includes(sp))who.push(sp);const c=COL[Math.max(who.indexOf(sp),0)%COL.length];
  return `<div class=row><span class=t>${esc(t)}</span><span class=who>${sp?`<b style="background:${c}"></b>${esc(sp)}`:''}</span><span class=x>${esc(tx)}</span></div>`}).join('')||'<div class=empty>No text.</div>';
}
$('jretry').onclick=async()=>{await api('/api/retry',{id:sel});lastJob=null;logFrom=0;$('log').textContent='';await tick()};
$('jcancel').onclick=async()=>{await api('/api/cancel',{id:sel});lastJob=null;await tick()};
$('jdel').onclick=()=>delJob(sel,lastJob&&(lastJob.title||lastJob.video_id));
$('jproj').onchange=async()=>{const v=$('jproj').value;await api('/api/move',{ids:[sel],project_id:v===''?null:+v});if(lastJob)lastJob.project_id=v===''?null:+v;await refresh()};
$('retryall').onclick=async()=>{await api('/api/retry',{all_failed:true});await loadList()};

/* ---- new transcript form ---- */
const params=()=>({start:+$('start').value,end:$('end').value===''?null:+$('end').value,model:$('model').value,speakers:$('spknames').value,label:$('label').checked});
const projParams=()=>{const v=$('proj').value;return v==='__new'?{new_project:$('newproj').value.trim()}:v===''?{}:{project_id:+v}};
async function enqueue(items){
 const p=params();if(p.end!=null&&p.end<=p.start)return {error:'End time must be after start time'};
 const pp=projParams();if('new_project' in pp&&!pp.new_project)return {error:'Type a name for the new project'};
 const {ok,d}=await api('/api/enqueue',{...p,...pp,items});if(!ok)return {error:d.error||'Failed'};
 if(d.project_id){await loadProjects();cur=d.project_id;$('proj').value=String(d.project_id);$('npw').hidden=true;$('newproj').value='';renderProjects()}
 return d;
}
$('go').onclick=async()=>{
 $('err').textContent=$('okmsg').textContent='';$('review').hidden=true;
 const p=params();if(p.end!=null&&p.end<=p.start)return $('err').textContent='End time must be after start time';
 $('go').disabled=true;
 try{
  if(mode=='one'){
   const r=await enqueue([{url:$('url').value.trim()}]);if(r.error)return $('err').textContent=r.error;
   $('url').value='';await refresh();const id=r.ids[0];if(id)openJob(id);else $('okmsg').textContent='Already in the queue.';
  }else{
   $('okmsg').textContent='Reading links...';
   const {ok,d}=await api('/api/expand',{text:$('urls').value});$('okmsg').textContent='';
   if(!ok)return $('err').textContent=d.error;
   if(!d.items.length)return $('err').textContent='No YouTube links found';
   if(d.expanded){review=d.items;suggest=d.suggest||'';$('rvtitle').textContent=`Found ${d.items.length} videos`+(suggest?` in “${suggest}”`:'');
    if(suggest&&$('proj').value===''){const ex=projects.find(x=>x.name.toLowerCase()==suggest.toLowerCase());if(ex)$('proj').value=String(ex.id);else{$('proj').value='__new';$('newproj').value=suggest}$('npw').hidden=$('proj').value!='__new'}
    $('rvlist').innerHTML=d.items.map((it,i)=>`<label><input type=checkbox checked data-i="${i}">${esc(it.title||it.id)}</label>`).join('');
    $('rverr').textContent='';$('rvgo').textContent=`Queue ${d.items.length} videos`;$('review').hidden=false;return}
   const r=await enqueue(d.items);if(r.error)return $('err').textContent=r.error;
   $('urls').value='';$('okmsg').textContent=`Queued ${r.queued} video${r.queued==1?'':'s'}`+(r.duplicates?` (${r.duplicates} already queued)`:'');await refresh();
  }
 }finally{$('go').disabled=false}
};
function picked(){return [...document.querySelectorAll('#rvlist input:checked')].map(e=>review[+e.dataset.i])}
$('rvlist').onchange=()=>{$('rvgo').textContent=`Queue ${picked().length} videos`};
$('rvall').onclick=()=>{const all=[...document.querySelectorAll('#rvlist input')],on=all.some(e=>!e.checked);all.forEach(e=>e.checked=on);$('rvall').textContent=on?'Untick all':'Tick all';$('rvlist').onchange()};
$('rvgo').onclick=async()=>{
 const items=picked();if(!items.length)return $('rverr').textContent='Tick at least one video';
 $('rvgo').disabled=true;const r=await enqueue(items);$('rvgo').disabled=false;if(r.error)return $('rverr').textContent=r.error;
 $('review').hidden=true;$('urls').value='';$('okmsg').textContent=`Queued ${r.queued} video${r.queued==1?'':'s'}`+(r.duplicates?` (${r.duplicates} already queued)`:'');await refresh();
};
init();
</script>
"""

if __name__ == "__main__":
    host, port = os.environ.get("UI_HOST", "127.0.0.1"), int(os.environ.get("UI_PORT", 8765))
    loopback = host in ("127.0.0.1", "localhost", "::1")
    db.init(DB_PATH)
    REQUIRE_LOGIN = (not loopback) or os.environ.get("UI_REQUIRE_LOGIN") == "1"  # behind a proxy the app only sees loopback, so the service sets the env flag
    if REQUIRE_LOGIN and db.count_users() == 0:
        sys.exit("login is required but there are no users yet. Create one first:  python3 manage.py adduser NAME")
    if not os.environ.get("NO_WORKER"): threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer((host, port), H)
    print(f"UI at http://{host}:{port}  (Ctrl+C to stop)  database {DB_PATH}")
    if loopback and not os.environ.get("NO_BROWSER"): threading.Timer(0.8, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    srv.serve_forever()
