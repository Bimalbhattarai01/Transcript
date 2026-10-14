#!/usr/bin/env python3
"""Team accounts for the web UI (run on the server, there is no signup page).

  python3 manage.py adduser alice       # asks for a password (min 10 chars)
  python3 manage.py passwd alice
  python3 manage.py disable alice       # / enable alice
  python3 manage.py deluser alice
  python3 manage.py listusers

On the server run it as the service user so the database file keeps its owner:
  sudo -u srt /opt/transcript/.venv/bin/python /opt/transcript/manage.py adduser alice
"""
import argparse, getpass, sys, time

import db


def ask_password():
    a = getpass.getpass("Password: ")
    if a != getpass.getpass("Repeat password: "): sys.exit("passwords do not match")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("adduser", "passwd", "disable", "enable", "deluser"): sub.add_parser(c).add_argument("username")
    sub.add_parser("listusers")
    a = ap.parse_args()
    db.init(db.default_path())
    try:
        if a.cmd == "adduser": db.add_user(a.username, ask_password()); print(f"created {a.username.lower()}")
        elif a.cmd == "passwd": db.set_password(a.username, ask_password()); print("password changed (that user is logged out everywhere)")
        elif a.cmd == "disable": db.set_disabled(a.username, True); print(f"{a.username} disabled")
        elif a.cmd == "enable": db.set_disabled(a.username, False); print(f"{a.username} enabled")
        elif a.cmd == "deluser": db.delete_user(a.username); print(f"{a.username} deleted")
        else:
            for u in db.list_users():
                seen = time.strftime("%Y-%m-%d %H:%M", time.localtime(u["last_login"])) if u["last_login"] else "never"
                print(f"{u['username']:<20} {'disabled' if u['disabled'] else 'active':<9} last login: {seen}")
            if not db.list_users(): print("no users yet: python3 manage.py adduser NAME")
    except ValueError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
