#!/bin/bash
# memrun.sh — run a command under a hard RSS watchdog (macOS-safe; RLIMIT_* is
# not reliably enforceable on Darwin, so we poll and kill).
# usage: scripts/memrun.sh <limit_mb> <command...>
set -u
LIMIT_MB=$1; shift
"$@" &
PID=$!
(
  while kill -0 "$PID" 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
    if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_MB * 1024)) ]; then
      echo "MEMGUARD: killing PID $PID (RSS $((RSS_KB / 1024)) MB > ${LIMIT_MB} MB limit)" >&2
      kill -9 "$PID"
      exit 137
    fi
    sleep 1
  done
) &
WATCHER=$!
wait "$PID"; RC=$?
kill "$WATCHER" 2>/dev/null
wait "$WATCHER" 2>/dev/null
exit $RC
