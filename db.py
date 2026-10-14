"""Tiny SQLite layer for the web UI: a job queue + history. One connection guarded by a lock (WAL mode)."""
import sqlite3, threading, time
from pathlib import Path

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
"""
LIST_COLS = ("id, batch_id, url, video_id, title, status, stage, progress, detail, start_s, end_s, model, label_speakers, "
             "cost, seconds, cues, words, error, created_at, started_at, finished_at")


def init(path):
    global _conn
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)  # autocommit
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL"); _conn.execute("PRAGMA busy_timeout=5000")
        _conn.executescript(SCHEMA)
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


def add_job(batch_id, url, video_id, title, start_s, end_s, model, speakers_hint, label_speakers):
    return x("INSERT INTO jobs(batch_id,url,video_id,title,start_s,end_s,model,speakers_hint,label_speakers,created_at) "
             "VALUES(?,?,?,?,?,?,?,?,?,?)",
             (batch_id, url, video_id, title, start_s, end_s, model, speakers_hint, int(bool(label_speakers)), time.time()))


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


def list_jobs(search="", limit=500):
    if search:
        like = f"%{search}%"
        return q(f"SELECT {LIST_COLS} FROM jobs WHERE title LIKE ? OR url LIKE ? OR video_id LIKE ? ORDER BY id DESC LIMIT ?",
                 (like, like, like, limit))
    return q(f"SELECT {LIST_COLS} FROM jobs ORDER BY id DESC LIMIT ?", (limit,))


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
