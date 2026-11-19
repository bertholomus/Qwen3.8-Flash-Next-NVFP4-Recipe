# For agents: install Qwen3.8-Flash-Next NVFP4 (GPTQ) on one RTX PRO 6000

This file is written for an AI agent (Claude Code, Codex, ChatGPT, Cursor, Hermes, ...) that installs this recipe for
a person. A person can also paste this one line into their agent:

> Install https://github.com/bertholomus/Qwen3.8-Flash-Next-NVFP4-Recipe on my RTX PRO 6000. Follow its AGENTS.md exactly.

## Rules for the agent

- **Ask before you act** for anything that needs root, changes a firewall, or stops a running model. Say what the
  command does first.
- **Never invent values.** Host addresses and paths come from the machines (`ip -br addr`, `df -h`) or from the
  person. If you cannot read them, ask.
- **Run every check in this file and show its output.** A step is done when its check passes, not when the command
  returns.
- **Keep the defaults** unless the person asks for something else. Section 2 says what each knob trades.
- **Do not publish numbers** from your run as this recipe's numbers. README.md holds the measured rows and on what
  hardware.
- If this file and the scripts disagree, follow the scripts and tell the person.

## Fast path (the whole install)

```bash
git clone https://github.com/bertholomus/Qwen3.8-Flash-Next-NVFP4-Recipe && cd Qwen3.8-Flash-Next-NVFP4-Recipe
cp scripts/local.sh.example scripts/local.sh   # optional: fill in only what you want to change
DRY_RUN=1 ./start.sh                           # prints the plan; changes nothing
./start.sh                                     # fetches the engine + weights, then serves
bash scripts/smoke.sh                          # exit 0 = every case passes
```

Then the API is `http://127.0.0.1:8080/v1`, model name `qwen3.8-flash-next-nvfp4`.

## 1. Check the machines

| check | command | expected |
|---|---|---|
| GPU | `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader` | one RTX PRO 6000, compute cap 12.0 (sm_120), ~95 GiB, driver 580+ (CUDA 13) |
| disk | `df -h "$HOME"` | ≥ 120 G free where the weights will go (download is 112.8 GB / 105.0 GiB, plus caches) |
| tools | `command -v bash curl tar sha256sum python3 git` | all found; `bash --version` 4 or newer |
| HF CLI | `hf version` | found, else `pip install -U "huggingface_hub[cli]"` |

If a check fails, fix it or tell the person before installing. Do not skip checks.

## 2. Pick the settings (every knob: values, default, trade)

Precedence, lowest to highest: `scripts/config.sh` defaults → `.env` (parsed as literal `KEY=value`, never sourced) →
`scripts/local.sh` → exported environment. Set a knob once, in the highest layer you need.

| knob | default | values | trade |
|---|---|---|---|
| `MODE` | `agents` | `single` / `agents` / `full4` | `agents`: `--parallel 8 --context 32768`, prompt cache on — most concurrency, 32K tokens per stream; the measured multi-agent rows are this mode. `single`: `--parallel 1 --context 262144` — the whole window on one stream. `full4` (EXPERIMENTAL): `--parallel 4 --context 262144` with `TF_FLASHNEXT_KV_PAGES=256` and the prompt cache off — four streams share one KV page pool, so each can reach the full window while the others are shorter; some requests wait for pages; only needle-finding is measured here (4/4 in 51–185 s), not throughput. |
| `PORT` | `8080` | any free port | change only when taken |
| `HOST` | `127.0.0.1` | loopback, or `0.0.0.0` | `0.0.0.0` puts the API on the network — only if the person asks; the Qwen licence has terms about hosting for others |
| `MODEL_DIR` | `$HOME/models/Qwen3.8-Flash-Next-NVFP4-GPTQ` | any absolute path | where the ~113 GB of weights land; `start.sh` runs `hf download` when it is empty |
| `TF_VERSION` | `1.2.9` | a released version | pinned engine release; change only together with `TF_SHA256` |
| `TF_SHA256` | `d17ba88f98d6e8b63141ed931c75db4561ed4ce7ba4ca7fbf8e28d98bbc27187` | 64 hex chars | sha256 of the release tarball; `start.sh` refuses a mismatch |
| `THINKING` | `off` | `off` / `on` | `off` starts with `--no-thinking` (the measured rows are thinking-off); `on` keeps the chat template's default |
| `NAME` | `qwen3.8-flash-next-nvfp4` | letters/digits/._- | the id clients send and `/v1/models` reports |

## 3. Install and boot (each step with its check)

1. Preflight — every section 1 check green. Check: show the outputs.
2. `DRY_RUN=1 ./start.sh`. Check: it prints every knob with its source layer, the tarball URL, the sha256 check and
   the exact serve command, ends with "nothing above was executed", and exits 0.
3. `./start.sh`. Check: lines `engine: …` and `weights: …` (a first run downloads ~113 GB), then
   `ready: http://127.0.0.1:8080/v1 (model: qwen3.8-flash-next-nvfp4)`.
4. `curl -fsS http://127.0.0.1:8080/v1/models`. Check: JSON listing `qwen3.8-flash-next-nvfp4`.
5. `bash scripts/smoke.sh`. Check: three `PASS` lines, then `ALL SMOKE CHECKS PASSED`, exit code 0.

## 4. Check that it serves

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-nvfp4","messages":[{"role":"user","content":"What is 17*19? Answer with the number only."}],"max_tokens":16,"temperature":0}'
```

Check: the reply content contains `323`.

## 5. When a step fails (table: what you see -> what to do)

| what you see | what to do |
|---|---|
| `sha256sum: … FAILED` | remove the tarball under `~/.cache/tensorfold/` and re-run `./start.sh`. Never skip the check. |
| `hf download` stalls or fails | check free disk (≥ 125 G) and network; re-run the same command — it resumes. |
| `not ready after 900s` | `tail -40 run/serve.log`. Usual causes: driver older than CUDA 13, another process holds the GPU (`nvidia-smi`), a leftover server holding memory. Then `./stop.sh` and re-run. |
| port already in use | set `PORT` in `scripts/local.sh` (or `.env`) and re-run. |
| startup prints a named GPU refusal | the release binary refuses cards it has no kernels for; confirm compute cap 12.0 (section 1). |
| a smoke check fails | confirm the server is the one this recipe started (`curl /v1/models`), then re-run once before digging. |

## 6. Not measured yet: tell the person

These numbers are not in the recipe — say so if the person asks:

- aggregate throughput of `full4` (only needle-finding is measured: 4/4 in 51–185 s);
- peak VRAM per mode and cold-load time;
- any GPU other than one RTX PRO 6000 Blackwell, or any engine other than TensorFold 1.2.9 on sm_120;
- contexts beyond 262,144 tokens (this recipe serves the native window; no YaRN window on CUDA).
