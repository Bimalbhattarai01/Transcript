"""Tiny .env loader (no dependency). Real environment variables win over .env."""
import os
from pathlib import Path


def load_env(path=None):
    p = Path(path) if path else Path(__file__).parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
