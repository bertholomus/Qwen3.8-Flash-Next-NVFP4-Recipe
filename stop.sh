#!/usr/bin/env bash
# stop.sh — stop the server started by start.sh (pid file in run/serve.pid).
set -u -o pipefail
cd "$(dirname "$0")"
# shellcheck source=scripts/config.sh
. scripts/config.sh

if [ ! -f "$PIDFILE" ]; then
  echo "stop: no pidfile at $PIDFILE — nothing this recipe started (is something still on port $PORT?)"
  exit 1
fi

cfg_pid="$(cat "$PIDFILE")"
if kill -0 "$cfg_pid" 2>/dev/null; then
  kill "$cfg_pid"
  cfg_i=0
  while [ "$cfg_i" -lt 30 ] && kill -0 "$cfg_pid" 2>/dev/null; do
    sleep 1
    cfg_i=$((cfg_i + 1))
  done
  if kill -0 "$cfg_pid" 2>/dev/null; then
    kill -9 "$cfg_pid" 2>/dev/null || true
    echo "stop: sent SIGKILL to pid $cfg_pid"
  else
    echo "stop: stopped (pid $cfg_pid)"
  fi
else
  echo "stop: pid $cfg_pid is not running (stale pidfile)"
fi
rm -f "$PIDFILE"
exit 0
