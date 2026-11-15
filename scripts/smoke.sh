#!/usr/bin/env bash
# scripts/smoke.sh — three checks against a running server; exit 0 only if all pass.
#   1. 17*19 -> 323            (basic arithmetic)
#   2. one tool call parses    (OpenAI tools shape, arguments are valid JSON)
#   3. a 16k-token needle      (retrieval inside a long prompt)
set -u -o pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/config.sh
. scripts/config.sh

BASE="$URL"
MODEL="$NAME"
PASS=0
FAIL=0

need() { command -v "$1" >/dev/null 2>&1 || { echo "smoke: missing dependency: $1" >&2; exit 2; }; }
need curl
need python3

result() { # result <n> <name> <0|1>
  if [ "$3" -eq 0 ]; then
    printf 'PASS %s: %s\n' "$1" "$2"
    PASS=$((PASS + 1))
  else
    printf 'FAIL %s: %s\n' "$1" "$2"
    FAIL=$((FAIL + 1))
  fi
}

# message content (or tool_calls JSON) of a saved response body
content_of() {
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
ch = d.get("choices") or [{}]
m = ch[0].get("message", {})
tc = m.get("tool_calls") or []
print(json.dumps(tc) if tc else (m.get("content") or ""))
' "$1"
}

chat() { # chat <body-file> -> response body on stdout
  curl -fsS --max-time 300 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary @"$1"
}

if ! curl -fsS --max-time 10 "$BASE/v1/models" >/dev/null 2>&1; then
  echo "smoke: no server at $BASE — run ./start.sh first" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ---- check 1: 17*19 -> 323
cat > "$TMP/c1.json" <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"What is 17 times 19? Answer with the number only."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}
EOF
if chat "$TMP/c1.json" >"$TMP/c1.out" 2>/dev/null; then
  C1="$(content_of "$TMP/c1.out" 2>/dev/null || true)"
else
  C1=""
fi
case "$C1" in
  *"323"*) result 1 "17*19 -> 323" 0 ;;
  *) result 1 "17*19 -> 323 (got: $(printf '%.60s' "$C1"))" 1 ;;
esac

# ---- check 2: one tool call parses
cat > "$TMP/c2.json" <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"Use the get_weather tool to look up Tokyo. Call the tool; do not answer in words."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}
EOF
cat > "$TMP/parse_tool.py" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = (d.get("choices") or [{}])[0].get("message", {})
tcs = m.get("tool_calls") or []
ok = False
if tcs:
    f = tcs[0].get("function", {})
    try:
        args = json.loads(f.get("arguments") or "")
        ok = f.get("name") == "get_weather" and isinstance(args, dict)
    except Exception:
        ok = False
sys.exit(0 if ok else 1)
PY
if chat "$TMP/c2.json" >"$TMP/c2.out" 2>/dev/null && python3 "$TMP/parse_tool.py" "$TMP/c2.out" 2>/dev/null; then
  result 2 "one tool call parses" 0
else
  result 2 "one tool call parses (no valid get_weather tool_call in reply)" 1
fi

# ---- check 3: 16k-token needle
cat > "$TMP/build_needle.py" <<'PY'
import json, sys
filler = ("Section {n}. The storage bay audit continues; every crate was weighed, logged, and "
          "shelved by the night crew, and the ledger was countersigned before dawn.")
doc = " ".join(filler.format(n=n) for n in range(1, 500))
while len(doc) < 62000:  # ~16k tokens at ~4 chars/token
    doc += " " + filler.format(n=500)
needle = " The vault codeword is PERIDOT-HARBOR-91. "
i = (len(doc) * 3) // 5
doc = doc[:i] + needle + doc[i:]
prompt = doc + "\n\nWhat is the vault codeword written inside the document above? Answer with the codeword only."
json.dump({"model": sys.argv[2],
           "messages": [{"role": "user", "content": prompt}],
           "max_tokens": 64, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}},
          open(sys.argv[1], "w"))
PY
python3 "$TMP/build_needle.py" "$TMP/c3.json" "$MODEL"
if chat "$TMP/c3.json" >"$TMP/c3.out" 2>/dev/null; then
  C3="$(content_of "$TMP/c3.out" 2>/dev/null || true)"
else
  C3=""
fi
case "$C3" in
  *"PERIDOT-HARBOR-91"*) result 3 "16k-token needle" 0 ;;
  *) result 3 "16k-token needle (got: $(printf '%.60s' "$C3"))" 1 ;;
esac

echo "smoke: $PASS passed, $FAIL failed (server $BASE, model $MODEL, MODE=$MODE)"
if [ "$FAIL" -eq 0 ]; then
  echo "ALL SMOKE CHECKS PASSED"
  exit 0
fi
exit 1
