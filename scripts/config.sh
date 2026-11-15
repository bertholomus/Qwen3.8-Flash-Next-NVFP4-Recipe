# scripts/config.sh — resolve every knob for the recipe scripts. Sourced by start.sh,
# stop.sh and scripts/smoke.sh. Requires bash 4+ (associative arrays).
#
# Layer order, lowest to highest:
#   1. defaults below (this file)
#   2. .env                 — parsed as literal KEY=value lines, NEVER sourced or executed
#   3. scripts/local.sh     — user overrides, sourced (copy scripts/local.sh.example)
#   4. exported environment — wins over everything above
#
# After sourcing, callers get MODE PORT HOST MODEL_DIR TF_VERSION TF_SHA256 THINKING NAME
# plus derived values (WEIGHTS_REPO, TF_TARBALL_URL, TF_ENGINE_DIR, SERVE_PARALLEL,
# SERVE_CONTEXT, FULL4_KV_PAGES, RUN_DIR, PIDFILE, LOGFILE, URL, WAIT_SECS) and the
# helpers cfg_plan (print every knob and where it came from), cfg_serve_argv and
# find_engine_bin.

cfg_die() { printf 'config: %s\n' "$*" >&2; exit 2; }

RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

declare -A CFG_VALUE=() CFG_FROM=() CFG_ENVSNAP=()
cfg_set() { CFG_VALUE[$1]=$2; CFG_FROM[$1]=$3; }

# --- layer 1: defaults
cfg_set MODE agents defaults
cfg_set PORT 8080 defaults
cfg_set HOST 127.0.0.1 defaults
cfg_set MODEL_DIR "$HOME/models/Qwen3.8-Flash-Next-NVFP4-GPTQ" defaults
cfg_set TF_VERSION 1.2.9 defaults
cfg_set TF_SHA256 d17ba88f98d6e8b63141ed931c75db4561ed4ce7ba4ca7fbf8e28d98bbc27187 defaults
cfg_set THINKING off defaults
cfg_set NAME qwen3.8-flash-next-nvfp4 defaults

CFG_KEYS=(MODE PORT HOST MODEL_DIR TF_VERSION TF_SHA256 THINKING NAME)

# snapshot the exported environment BEFORE scripts/local.sh can shadow it (layer 4 wins)
for cfg_k in "${CFG_KEYS[@]}"; do
  if [ -n "${!cfg_k+x}" ]; then CFG_ENVSNAP[$cfg_k]=${!cfg_k}; fi
done

# --- layer 2: .env, parsed as literal KEY=value lines (never sourced)
ENV_FILE="$RECIPE_ROOT/.env"
if [ -f "$ENV_FILE" ]; then
  while IFS= read -r cfg_line || [ -n "$cfg_line" ]; do
    cfg_line=${cfg_line%$'\r'}
    case $cfg_line in ''|\#*) continue ;; esac
    case $cfg_line in 'export '*) cfg_line=${cfg_line#'export '} ;; esac
    case $cfg_line in
      *=*) cfg_key=${cfg_line%%=*}; cfg_val=${cfg_line#*=} ;;
      *) cfg_die ".env: not KEY=value: $cfg_line" ;;
    esac
    cfg_key="${cfg_key#"${cfg_key%%[![:space:]]*}"}"; cfg_key="${cfg_key%"${cfg_key##*[![:space:]]}"}"
    cfg_val="${cfg_val#"${cfg_val%%[![:space:]]*}"}"; cfg_val="${cfg_val%"${cfg_val##*[![:space:]]}"}"
    if [ "${#cfg_val}" -ge 2 ] && [ "${cfg_val:0:1}" = '"' ] && [ "${cfg_val: -1}" = '"' ]; then
      cfg_val=${cfg_val:1:${#cfg_val}-2}   # strip one matching pair of double quotes; rest is literal
    fi
    case $cfg_key in
      MODE|PORT|HOST|MODEL_DIR|TF_VERSION|TF_SHA256|THINKING|NAME)
        cfg_set "$cfg_key" "$cfg_val" ".env" ;;
      *) cfg_die ".env: unknown key '$cfg_key' (allowed: MODE PORT HOST MODEL_DIR TF_VERSION TF_SHA256 THINKING NAME)" ;;
    esac
  done < "$ENV_FILE"
fi
unset cfg_line cfg_key cfg_val ENV_FILE

# --- layer 3 (scripts/local.sh) merged under layer 4 (exported env)
LOCAL_SH="$RECIPE_ROOT/scripts/local.sh"
if [ -f "$LOCAL_SH" ]; then
  # shellcheck source=/dev/null
  . "$LOCAL_SH"
fi
for cfg_k in "${CFG_KEYS[@]}"; do
  if [ -n "${CFG_ENVSNAP[$cfg_k]+x}" ]; then
    cfg_set "$cfg_k" "${CFG_ENVSNAP[$cfg_k]}" "exported env"
  elif [ -n "${!cfg_k+x}" ]; then
    cfg_set "$cfg_k" "${!cfg_k}" "scripts/local.sh"
  fi
done
unset cfg_k LOCAL_SH

# --- resolve + validate
MODE=${CFG_VALUE[MODE]}; PORT=${CFG_VALUE[PORT]}; HOST=${CFG_VALUE[HOST]}
MODEL_DIR=${CFG_VALUE[MODEL_DIR]}; TF_VERSION=${CFG_VALUE[TF_VERSION]}
TF_SHA256=${CFG_VALUE[TF_SHA256]}; THINKING=${CFG_VALUE[THINKING]}; NAME=${CFG_VALUE[NAME]}

case $MODE in single|agents|full4) ;; *) cfg_die "MODE must be single, agents or full4 (got '$MODE')" ;; esac
case $THINKING in on|off) ;; *) cfg_die "THINKING must be on or off (got '$THINKING')" ;; esac
case $PORT in ''|*[!0-9]*) cfg_die "PORT must be a number (got '$PORT')" ;; esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || cfg_die "PORT out of range: $PORT"
[ -n "$HOST" ] || cfg_die "HOST must not be empty"
case $NAME in ''|*[!A-Za-z0-9._-]*) cfg_die "NAME may contain only letters, digits, '.', '_' and '-' (got '$NAME')" ;; esac
case $TF_VERSION in ''|*[!0-9.]*) cfg_die "TF_VERSION must look like 1.2.9 (got '$TF_VERSION')" ;; esac
case $TF_SHA256 in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
  *) cfg_die "TF_SHA256 must be 64 lowercase hex chars" ;;
esac
[ "${#TF_SHA256}" -eq 64 ] || cfg_die "TF_SHA256 must be 64 hex chars (got ${#TF_SHA256})"
case $MODEL_DIR in /*) ;; *) cfg_die "MODEL_DIR must be an absolute path (got '$MODEL_DIR')" ;; esac

# --- derived
WEIGHTS_REPO=bertholomus/Qwen3.8-Flash-Next-NVFP4-GPTQ
TF_TARBALL="tensorfold-${TF_VERSION}-linux-x86_64.tar.gz"
TF_TARBALL_URL="https://github.com/ashhart/TensorFold/releases/download/v${TF_VERSION}/${TF_TARBALL}"
TF_CACHE_DIR="${TENSORFOLD_HOME:-$HOME/.cache/tensorfold}"
TF_ENGINE_DIR="$TF_CACHE_DIR/tensorfold-${TF_VERSION}"
RUN_DIR="$RECIPE_ROOT/run"
PIDFILE="$RUN_DIR/serve.pid"
LOGFILE="$RUN_DIR/serve.log"
URL="http://$HOST:$PORT"
WAIT_SECS="${WAIT_SECS:-900}"
FULL4_KV_PAGES=256
case $MODE in
  single) SERVE_PARALLEL=1; SERVE_CONTEXT=262144 ;;
  agents) SERVE_PARALLEL=8; SERVE_CONTEXT=32768 ;;
  full4)  SERVE_PARALLEL=4; SERVE_CONTEXT=262144 ;;
esac

# path of the downloaded release binary (tensorfold-native preferred)
find_engine_bin() {
  local cfg_c
  for cfg_c in "$TF_ENGINE_DIR/tensorfold-native" "$TF_ENGINE_DIR/bin/tensorfold-native" \
               "$TF_ENGINE_DIR/tensorfold" "$TF_ENGINE_DIR/bin/tensorfold"; do
    [ -x "$cfg_c" ] && { printf '%s\n' "$cfg_c"; return 0; }
  done
  cfg_c="$(find "$TF_ENGINE_DIR" -maxdepth 3 -type f \( -name tensorfold-native -o -name tensorfold \) 2>/dev/null | head -n 1)"
  [ -n "$cfg_c" ] && { printf '%s\n' "$cfg_c"; return 0; }
  return 1
}

# exact serve argv (shell-quoted) for a given engine binary path
cfg_serve_argv() {
  local cfg_bin=$1
  local cfg_a=(serve "$MODEL_DIR" --host "$HOST" --port "$PORT" --parallel "$SERVE_PARALLEL" --context "$SERVE_CONTEXT" --name "$NAME")
  [ "$THINKING" = off ] && cfg_a+=(--no-thinking)
  [ "$MODE" = full4 ] && cfg_a+=(--prompt-cache-gib 0)
  local cfg_out="" cfg_x
  for cfg_x in "$cfg_bin" "${cfg_a[@]}"; do cfg_out+="$(printf '%q ' "$cfg_x")"; done
  printf '%s' "${cfg_out% }"
}

# every knob with the layer it came from (DRY_RUN and debugging)
cfg_plan() {
  local cfg_k
  printf 'resolved settings (source in parentheses)\n'
  for cfg_k in "${CFG_KEYS[@]}"; do
    printf '  %-12s %s  (%s)\n' "$cfg_k" "${CFG_VALUE[$cfg_k]}" "${CFG_FROM[$cfg_k]}"
  done
  printf '  %-12s %s  (derived)\n' 'mode-flags' "--parallel $SERVE_PARALLEL --context $SERVE_CONTEXT"
  if [ "$MODE" = full4 ]; then
    printf '  %-12s %s  (derived)\n' 'full4-env' "TF_FLASHNEXT_KV_PAGES=$FULL4_KV_PAGES; prompt cache off (--prompt-cache-gib 0)"
  fi
  printf '  %-12s %s  (derived)\n' 'engine-dir' "$TF_ENGINE_DIR"
  printf '  %-12s %s  (derived)\n' 'tarball' "$TF_TARBALL_URL"
  printf '  %-12s %s  (derived)\n' 'weights' "$WEIGHTS_REPO -> $MODEL_DIR"
  printf '  %-12s %s  (derived)\n' 'url' "$URL/v1"
}
