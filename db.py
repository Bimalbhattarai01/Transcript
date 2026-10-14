"""Tiny SQLite layer for the web UI: a job queue + history. One connection guarded by a lock (WAL mode)."""
import os, re, sqlite3, threading, time
from pathlib import Path

import auth
from env import load_env

_lock = threading.RLock()
_conn = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id INTEGER, url TEXT NOT NULL, video_id TEXT, title TEXT,
    status TEXT NOT NULL DEFAULT 'queued',            -- queued | running | done | error | cancelled
    stage TEXT DEFAULT 'queued', progress REAL DEFAULT 0, detail TEXT DEFAULT '',
    start_s REAL DEFAULT 0, end_s REAL, model TEXT NOT NULL, speakers_hint TEXT DEFAULT '', label_speakers INTEGER DEFAULT 1,
    cost REAL, seconds REAL, cues INTEGER, words INTEGER, srt_text TEXT, error TEXT, log TEXT DEFAULT '',
    created_at REAL, started_at REAL, finished_at REAL);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, id);
CREATE INDEX IF NOT EXISTS jobs_video ON jobs(video_id);
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE COLLATE NOCASE, created_by TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE COLLATE NOCASE, pw_hash TEXT NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0, created_at REAL, last_login REAL);
"""
LIST_COLS = ("id, batch_id, url, video_id, title, status, stage, progress, detail, start_s, end_s, model, label_speakers, "
             "cost, seconds, cues, words, error, created_by, project_id, created_at, started_at, finished_at")


def default_path():
    """data/app.db under DATA_DIR (default: the repo folder); the server and manage.py must agree on it."""
    load_env()
    return Path(os.environ.get("DATA_DIR") or Path(__file__).parent) / "data" / "app.db"


def init(path):
    global _conn
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)  # autocommit
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL"); _conn.execute("PRAGMA busy_timeout=5000")
        _conn.executescript(SCHEMA)
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(jobs)")}
        for col, typ in (("created_by", "TEXT"), ("project_id", "INTEGER")):  # databases made before accounts/projects existed
            if col not in cols: _conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
        _conn.execute("CREATE INDEX IF NOT EXISTS jobs_project ON jobs(project_id)")
        # a crash/restart mid-job: put it back in the queue (chunk + diarization caches make the retry cheap)
        _conn.execute("UPDATE jobs SET status='queued', stage='queued', progress=0, detail='' WHERE status='running'")


def q(sql, args=()):
    with _lock:
        return [dict(r) for r in _conn.execute(sql, args).fetchall()]


def x(sql, args=()):
    with _lock:
        return _conn.execute(sql, args).lastrowid


def add_batch(name):
    return x("INSERT INTO batches(name, created_at) VALUES(?,?)", (name, time.time()))


def add_job(batch_id, url, video_id, title, start_s, end_s, model, speakers_hint, label_speakers, created_by=None, project_id=None):
    return x("INSERT INTO jobs(batch_id,url,video_id,title,start_s,end_s,model,speakers_hint,label_speakers,created_by,project_id,created_at) "
             "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
             (batch_id, url, video_id, title, start_s, end_s, model, speakers_hint, int(bool(label_speakers)), created_by, project_id,
              time.time()))


def get(jid, cols="*"):
    r = q(f"SELECT {cols} FROM jobs WHERE id=?", (jid,))
    return r[0] if r else None


def update(jid, **kw):
    if not kw: return
    x(f"UPDATE jobs SET {', '.join(k + '=?' for k in kw)} WHERE id=?", (*kw.values(), jid))


def append_log(jid, line):
    x("UPDATE jobs SET log = log || ? WHERE id=?", (line + "\n", jid))


def next_queued():
    r = q("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1")
    return r[0] if r else None


def _scope(project):
    """SQL condition for a project filter: None = everything, 0 = no project, n = that project."""
    if project is None: return "", ()
    return ("project_id IS NULL", ()) if project == 0 else ("project_id=?", (project,))


def list_jobs(search="", project=None, limit=500):
    conds, args = [], []
    if search:
        like = f"%{search}%"; conds.append("(title LIKE ? OR url LIKE ? OR video_id LIKE ?)"); args += [like, like, like]
    c, a2 = _scope(project)
    if c: conds.append(c); args += list(a2)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    return q(f"SELECT {LIST_COLS} FROM jobs {where} ORDER BY id DESC LIMIT ?", (*args, limit))


def delete_jobs(ids):
    """Delete finished/failed/queued jobs (never a running one). Returns how many were removed."""
    ids = [int(i) for i in ids]
    if not ids: return 0
    with _lock:
        n = _conn.execute(f"DELETE FROM jobs WHERE status<>'running' AND id IN ({','.join('?' * len(ids))})", ids).rowcount
    return n


def move_jobs(ids, project_id):
    ids = [int(i) for i in ids]
    if ids: x(f"UPDATE jobs SET project_id=? WHERE id IN ({','.join('?' * len(ids))})", (project_id, *ids))


# ---------- projects (folders for jobs, e.g. one per channel) ----------
def _clean_project(name):
    name = " ".join((name or "").split())
    if not 1 <= len(name) <= 60: raise ValueError("project name: 1-60 characters")
    return name


def add_project(name, created_by=None):
    name = _clean_project(name)
    r = q("SELECT id FROM projects WHERE name=?", (name,))
    if r: raise ValueError(f"a project called '{name}' already exists")
    return x("INSERT INTO projects(name, created_by, created_at) VALUES(?,?,?)", (name, created_by, time.time()))


def rename_project(pid, name):
    name = _clean_project(name)
    r = q("SELECT id FROM projects WHERE name=? AND id<>?", (name, pid))
    if r: raise ValueError(f"a project called '{name}' already exists")
    x("UPDATE projects SET name=? WHERE id=?", (name, pid))


def delete_project(pid, delete_jobs_too=False):
    if delete_jobs_too:
        ids = [r["id"] for r in q("SELECT id FROM jobs WHERE project_id=?", (pid,))]
        delete_jobs(ids)
    x("UPDATE jobs SET project_id=NULL WHERE project_id=?", (pid,))  # anything left (e.g. a running job) is un-filed, not lost
    x("DELETE FROM projects WHERE id=?", (pid,))


def list_projects():
    return q("SELECT p.id, p.name, COUNT(j.id) total, COALESCE(SUM(j.status='done'), 0) done FROM projects p "
             "LEFT JOIN jobs j ON j.project_id=p.id GROUP BY p.id ORDER BY p.name COLLATE NOCASE")


def project_exists(pid):
    return bool(q("SELECT 1 FROM projects WHERE id=?", (pid,)))


def unfiled_count():
    return q("SELECT COUNT(*) n FROM jobs WHERE project_id IS NULL")[0]["n"]


def counts():
    c = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) n FROM jobs GROUP BY status")}
    return {k: c.get(k, 0) for k in ("queued", "running", "done", "error", "cancelled")}


def find_done(video_id, start_s, end_s, model, speakers_hint, label_speakers, exclude_id):
    """An earlier finished job with identical settings (its SRT can be reused for free)."""
    r = q("SELECT * FROM jobs WHERE status='done' AND video_id=? AND start_s=? AND COALESCE(end_s,-1)=COALESCE(?,-1) AND model=? "
          "AND speakers_hint=? AND label_speakers=? AND id<>? AND srt_text IS NOT NULL ORDER BY id DESC LIMIT 1",
          (video_id, start_s, end_s, model, speakers_hint, int(bool(label_speakers)), exclude_id))
    return r[0] if r else None


def active_duplicate(video_id, start_s, end_s, model, speakers_hint, label_speakers):
    r = q("SELECT id FROM jobs WHERE status IN ('queued','running') AND video_id=? AND start_s=? AND COALESCE(end_s,-1)=COALESCE(?,-1) "
          "AND model=? AND speakers_hint=? AND label_speakers=? LIMIT 1",
          (video_id, start_s, end_s, model, speakers_hint, int(bool(label_speakers))))
    return r[0]["id"] if r else None


# ---------- team accounts ----------
USERNAME = re.compile(r"[a-z0-9_.@+-]{2,64}")  # plain names or email addresses


def add_user(username, password):
    username = username.strip().lower()
    if not USERNAME.fullmatch(username): raise ValueError("username: 2-64 characters, only a-z 0-9 _ . @ + - (an email address works)")
    if len(password) < auth.MIN_LEN: raise ValueError(f"password must be at least {auth.MIN_LEN} characters")
    if get_user(username): raise ValueError(f"user '{username}' already exists")
    x("INSERT INTO users(username, pw_hash, created_at) VALUES(?,?,?)", (username, auth.hash_password(password), time.time()))


def get_user(username):
    r = q("SELECT * FROM users WHERE username=?", (username.strip().lower(),))
    return r[0] if r else None


def set_password(username, password):
    if len(password) < auth.MIN_LEN: raise ValueError(f"password must be at least {auth.MIN_LEN} characters")
    if not get_user(username): raise ValueError(f"no such user '{username}'")
    x("UPDATE users SET pw_hash=? WHERE username=?", (auth.hash_password(password), username.strip().lower()))


def set_disabled(username, disabled):
    if not get_user(username): raise ValueError(f"no such user '{username}'")
    x("UPDATE users SET disabled=? WHERE username=?", (int(disabled), username.strip().lower()))


def delete_user(username):
    if not get_user(username): raise ValueError(f"no such user '{username}'")
    x("DELETE FROM users WHERE username=?", (username.strip().lower(),))


def list_users():
    return q("SELECT username, disabled, created_at, last_login FROM users ORDER BY username")


def touch_login(username):
    x("UPDATE users SET last_login=? WHERE username=?", (time.time(), username))


def count_users():
    return q("SELECT COUNT(*) n FROM users")[0]["n"]
