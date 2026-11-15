#!/usr/bin/env bash
# start.sh — download the stock TensorFold release binary (sha256-checked), download the
# weights with `hf download` if MODEL_DIR is empty, serve, wait for /v1/models, print the URL.
# DRY_RUN=1 ./start.sh prints the full plan and changes nothing.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=scripts/config.sh
. scripts/config.sh

say() { printf '%s\n' "$*"; }

if [ "${DRY_RUN:-0}" = "1" ]; then
  cfg_plan
  say ""
  say "planned steps:"
  if cfg_bin="$(find_engine_bin)"; then
    say "  1. engine: already present at $cfg_bin (no download)"
  else
    say "  1. engine: curl -fL --retry 3 -o $TF_CACHE_DIR/$TF_TARBALL.part \\"
    say "         $TF_TARBALL_URL"
    say "             echo \"$TF_SHA256  \$tarball\" | sha256sum -c -   # must print OK"
    say "             mkdir -p $TF_ENGINE_DIR && tar -xzf \$tarball -C $TF_ENGINE_DIR"
  fi
  if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    say "  2. weights: already present at $MODEL_DIR (no download)"
  else
    say "  2. weights: hf download $WEIGHTS_REPO --local-dir $MODEL_DIR"
  fi
  say "  3. mkdir -p $RUN_DIR"
  say "  4. serve (log $LOGFILE, pid $PIDFILE):"
  if [ "$MODE" = full4 ]; then
    say "       env TF_FLASHNEXT_KV_PAGES=$FULL4_KV_PAGES \\"
  fi
  cfg_flags=""
  [ "$THINKING" = off ] && cfg_flags="$cfg_flags --no-thinking"
  [ "$MODE" = full4 ] && cfg_flags="$cfg_flags --prompt-cache-gib 0"
  say "       <engine binary> serve $MODEL_DIR --host $HOST --port $PORT --parallel $SERVE_PARALLEL --context $SERVE_CONTEXT --name $NAME$cfg_flags"
  say "  5. wait up to ${WAIT_SECS}s for GET $URL/v1/models, then print the URL"
  say ""
  say "DRY_RUN=1: nothing above was executed."
  exit 0
fi

for cfg_bindep in curl tar sha256sum; do
  command -v "$cfg_bindep" >/dev/null 2>&1 || cfg_die "missing dependency: $cfg_bindep"
done

cfg_bin="$(find_engine_bin || true)"
if [ -z "$cfg_bin" ]; then
  say "engine: downloading $TF_TARBALL_URL"
  mkdir -p "$TF_CACHE_DIR"
  curl -fL --retry 3 --progress-bar -o "$TF_CACHE_DIR/$TF_TARBALL.part" "$TF_TARBALL_URL"
  mv -f "$TF_CACHE_DIR/$TF_TARBALL.part" "$TF_CACHE_DIR/$TF_TARBALL"
  printf '%s  %s\n' "$TF_SHA256" "$TF_CACHE_DIR/$TF_TARBALL" | sha256sum -c -
  mkdir -p "$TF_ENGINE_DIR"
  tar -xzf "$TF_CACHE_DIR/$TF_TARBALL" -C "$TF_ENGINE_DIR"
  cfg_bin="$(find_engine_bin)" || cfg_die "no tensorfold executable found in $TF_ENGINE_DIR"
  chmod +x "$cfg_bin"
fi
say "engine: $cfg_bin"

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
  command -v hf >/dev/null 2>&1 || cfg_die "hf CLI not found — pip install -U \"huggingface_hub[cli]\""
  say "weights: downloading $WEIGHTS_REPO into $MODEL_DIR"
  hf download "$WEIGHTS_REPO" --local-dir "$MODEL_DIR"
else
  say "weights: present at $MODEL_DIR"
fi
[ -f "$MODEL_DIR/model.safetensors.index.json" ] \
  || cfg_die "$MODEL_DIR has no model.safetensors.index.json — not a usable weights dir"

mkdir -p "$RUN_DIR"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  say "already running (pid $(cat "$PIDFILE")): $URL/v1 (model: $NAME)"
  exit 0
fi

declare -a SERVE_ARGS=(serve "$MODEL_DIR" --host "$HOST" --port "$PORT" \
                       --parallel "$SERVE_PARALLEL" --context "$SERVE_CONTEXT" --name "$NAME")
[ "$THINKING" = off ] && SERVE_ARGS+=(--no-thinking)
if [ "$MODE" = full4 ]; then
  SERVE_ARGS+=(--prompt-cache-gib 0)
  say "serve: TF_FLASHNEXT_KV_PAGES=$FULL4_KV_PAGES $(cfg_serve_argv "$cfg_bin") (prompt cache off)"
  nohup env TF_FLASHNEXT_KV_PAGES="$FULL4_KV_PAGES" "$cfg_bin" "${SERVE_ARGS[@]}" >>"$LOGFILE" 2>&1 &
else
  say "serve: $(cfg_serve_argv "$cfg_bin")"
  nohup "$cfg_bin" "${SERVE_ARGS[@]}" >>"$LOGFILE" 2>&1 &
fi
echo $! > "$PIDFILE"

cfg_wait=0
say "waiting up to ${WAIT_SECS}s for $URL/v1/models ..."
while [ "$cfg_wait" -lt "$WAIT_SECS" ]; do
  if curl -fsS --max-time 5 "$URL/v1/models" >/dev/null 2>&1; then
    say "ready: $URL/v1 (model: $NAME)"
    say "stop: ./stop.sh    logs: $LOGFILE"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    say "server exited during startup — last log lines:" >&2
    tail -n 20 "$LOGFILE" >&2 || true
    exit 1
  fi
  sleep 2
  cfg_wait=$((cfg_wait + 2))
done
say "not ready after ${WAIT_SECS}s — last log lines:" >&2
tail -n 20 "$LOGFILE" >&2 || true
exit 1
