#!/usr/bin/env bash
# One-shot Docker deploy. Run on the server from the cloned project folder:
#   cd /opt/transcript && bash deploy/vps-docker.sh
# It only touches this folder and one compose project (container transcribe-srt + volume srt-data).
# Re-run it any time to update after a `git pull`. DRY_RUN=1 stops after the checks.
set -euo pipefail
cd "$(dirname "$0")/.."
say() { printf '\n== %s\n' "$*"; }
die() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

say "1/5 checks"
command -v docker >/dev/null || die "docker is not installed"
docker compose version >/dev/null 2>&1 || die "'docker compose' (v2 plugin) is not available"
[ -f docker-compose.yml ] || die "run this from the project folder (docker-compose.yml not found)"
PORT="$(grep -s '^HOST_PORT=' .env | cut -d= -f2 || true)"; PORT="${PORT:-8765}"
if ss -ltn 2>/dev/null | grep -q ":$PORT " && ! docker ps --format '{{.Names}}' | grep -qx transcribe-srt; then
    die "port $PORT is already used by something else. Put HOST_PORT=<free port> in .env and run again. (Nothing was changed.)"
fi
free_kb="$(df -Pk . | awk 'NR==2{print $4}')"
[ "${free_kb:-0}" -ge 3145728 ] || die "less than 3 GB free on this disk ($((free_kb / 1024)) MB). The image needs about 1.2 GB plus room to build."
echo "docker ok, port $PORT free, $((free_kb / 1024)) MB free"

say "2/5 settings (.env)"
[ -f .env ] || { cp .env.example .env; chmod 600 .env; echo "created .env"; }
if ! grep -qE '^OPENROUTER_API_KEY=.{10,}' .env || grep -q 'your_openrouter_key_here' .env; then
    read -rsp "OpenRouter API key (typing is hidden): " key; echo
    [ -n "$key" ] || die "no key entered"
    grep -v '^OPENROUTER_API_KEY=' .env > .env.tmp || true
    printf 'OPENROUTER_API_KEY=%s\n' "$key" >> .env.tmp
    mv .env.tmp .env; chmod 600 .env; unset key
fi
echo "OpenRouter key is set"
[ "${DRY_RUN:-0}" = 1 ] && { echo "DRY_RUN: stopping before build."; exit 0; }

say "3/5 build the image (a few minutes the first time)"
docker compose build

say "4/5 team login"
n="$(docker compose run --rm -T srt python -c 'import db; db.init(db.default_path()); print(db.count_users())' 2>/dev/null | tail -n1 || true)"
if [ "${n:-0}" = "0" ]; then
    echo "No user exists yet. The app refuses to start without one, so create the first login now."
    read -rp "Email (this is the login name): " email
    docker compose run --rm srt python manage.py adduser "$email"
else
    echo "$n user(s) already exist"
fi

say "5/5 start"
docker compose up -d
BIND="$(grep -s '^BIND_ADDR=' .env | cut -d= -f2 || true)"; BIND="${BIND:-127.0.0.1}"
for i in $(seq 1 60); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://$BIND:$PORT/login" || true)"
    [ "$code" = 200 ] && break
    sleep 3
done
if [ "$code" != 200 ]; then
    docker compose logs --tail 40 srt
    die "the app did not come up (last HTTP code: ${code:-none}). The log above shows why."
fi
IP="$(curl -4 -s --max-time 5 ifconfig.me || hostname -I | awk '{print $1}')"
cat <<MSG

Done. The app is running and listening on $BIND:$PORT only (not reachable from the internet).

Open it from YOUR computer through an SSH tunnel (leave that terminal open):
    ssh -L $PORT:127.0.0.1:$PORT $(whoami)@$IP
then browse  http://127.0.0.1:$PORT   and sign in with the email you just created.

Useful:
    docker compose logs -f srt                                   # live log
    docker compose run --rm srt python manage.py adduser NAME    # add a teammate (also: listusers, passwd, disable, deluser)
    git pull && bash deploy/vps-docker.sh                        # update
MSG
