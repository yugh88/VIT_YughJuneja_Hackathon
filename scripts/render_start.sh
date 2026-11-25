#!/bin/sh
# Start Sentinel on Render and keep its memory in check (free tier: 512 MB).
#
# Two guards, both restart the server gracefully, which clears every in-memory buffer; the
# replay demo resumes on its own in about 10 s:
#   1. Memory: every SENTINEL_MEM_CHECK_S (15 s) the memory in use is read (see mem_mb); above
#      SENTINEL_MEM_LIMIT_MB (440 MB) it is restarted before it can run out.
#   2. Timer: every SENTINEL_RESTART_EVERY_S (25 h) it is restarted anyway, so memory never
#      builds up over a long run. Set it to 0 to turn the timer off.
LIMIT_MB="${SENTINEL_MEM_LIMIT_MB:-440}"
CHECK_S="${SENTINEL_MEM_CHECK_S:-15}"
EVERY_S="${SENTINEL_RESTART_EVERY_S:-90000}"
PORT="${PORT:-10000}"
pid=""

# Memory that counts toward the limit: the container's anonymous memory (cgroup v2), else the
# server's own RssAnon. VmRSS is not used: it also counts shared library pages (~100 MB here).
mem_mb() {
  m=""
  [ -r /sys/fs/cgroup/memory.stat ] && m=$(awk '/^anon / {print int($2 / 1048576)}' /sys/fs/cgroup/memory.stat)
  [ -n "$m" ] || m=$(awk '/^RssAnon/ {print int($2 / 1024)}' "/proc/$pid/status" 2>/dev/null)
  echo "${m:-0}"
}

stop() { [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; exit 0; }
trap stop TERM INT

while true; do
  uvicorn engine.api.app:app --host 0.0.0.0 --port "$PORT" &
  pid=$!
  started=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$CHECK_S" &
    wait $!
    rss=$(mem_mb)
    up=$(( $(date +%s) - started ))
    if [ "${rss:-0}" -gt "$LIMIT_MB" ]; then
      echo "watchdog: memory ${rss} MB > ${LIMIT_MB} MB, restarting the server to free it"
      kill -TERM "$pid" 2>/dev/null
      break
    elif [ "$EVERY_S" -gt 0 ] && [ "$up" -ge "$EVERY_S" ]; then
      echo "watchdog: scheduled restart after ${up} s (memory ${rss:-?} MB)"
      kill -TERM "$pid" 2>/dev/null
      break
    fi
  done
  wait "$pid" 2>/dev/null
  echo "watchdog: starting the server again"
  sleep 1
done
