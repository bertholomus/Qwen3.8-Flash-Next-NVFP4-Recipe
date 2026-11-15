# Qwen3.8-Flash-Next NVFP4 (GPTQ experts) — TensorFold recipe for one RTX PRO 6000

> Install https://github.com/bertholomus/Qwen3.8-Flash-Next-NVFP4-Recipe on my RTX PRO 6000. Follow its AGENTS.md exactly.

**Weights:** [bertholomus/Qwen3.8-Flash-Next-NVFP4-GPTQ](https://huggingface.co/bertholomus/Qwen3.8-Flash-Next-NVFP4-GPTQ)
— GPTQ-calibrated NVFP4 routed experts (ModelOpt `MIXED_PRECISION` layout), BF16 dense parts, n-gram table as NVFP4 rows.
**Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) @ `812d96e10a0dd82b69d749e68693698cf6c36daa`.
**Licence:** the weights are © Qwen under the Qwen Community License 1.0 (see [Licence](#licence)).
**Engine:** stock [TensorFold](https://github.com/ashhart/TensorFold) **1.2.9** release binary (Apache-2.0), downloaded and
sha256-checked by `start.sh`. No fork, no patches. Not affiliated with or endorsed by Qwen or TensorFold.

## Hardware

- One **RTX PRO 6000 Blackwell, 96 GB** (compute capability `sm_120`). This is the only configuration measured.
- Linux x86_64, GPU driver for **CUDA 13 or newer**.
- ~115 GB free disk (weights 109.1 GB, engine ~60 MB, caches).
- `bash` 4+, `python3`, `curl`, `tar`, and the HF CLI for the weight download: `pip install -U "huggingface_hub[cli]"`.

## Quick start

```bash
git clone https://github.com/bertholomus/Qwen3.8-Flash-Next-NVFP4-Recipe
cd Qwen3.8-Flash-Next-NVFP4-Recipe
DRY_RUN=1 ./start.sh     # prints the full plan; changes nothing
./start.sh               # downloads engine + weights, serves, waits for /v1/models, prints the URL
bash scripts/smoke.sh    # 3 checks; exit 0 = all pass
./stop.sh
```

The API is `http://127.0.0.1:8080/v1` (OpenAI-compatible), model name `qwen3.8-flash-next-nvfp4`.
AI agents: read [AGENTS.md](AGENTS.md) — it is the install runbook with a check after every step.

## Modes

| MODE | serve flags | notes |
|---|---|---|
| `agents` (default) | `--parallel 8 --context 32768` | up to 8 concurrent agent streams; prompt cache on |
| `single` | `--parallel 1 --context 262144` | one stream at the full 262,144-token window |
| `full4` **(EXPERIMENTAL)** | `--parallel 4 --context 262144`, `TF_FLASHNEXT_KV_PAGES=256`, prompt cache off (`--prompt-cache-gib 0`) | four streams share one KV page pool; each can reach the full window when the others are shorter. TensorFold marks this mode experimental. |

## Settings

Precedence, lowest to highest: `scripts/config.sh` defaults → `.env` → `scripts/local.sh` (copy from
`scripts/local.sh.example`) → exported environment. `.env` is parsed as literal `KEY=value` lines and is
**never sourced** — the scripts read it as data.

| knob | default | meaning |
|---|---|---|
| `MODE` | `agents` | `single`, `agents`, `full4` (table above) |
| `PORT` | `8080` | HTTP port |
| `HOST` | `127.0.0.1` | bind address (`0.0.0.0` exposes the server — mind the licence if you host for others) |
| `MODEL_DIR` | `$HOME/models/Qwen3.8-Flash-Next-NVFP4-GPTQ` | weights dir; `start.sh` runs `hf download` into it when it is empty |
| `TF_VERSION` | `1.2.9` | TensorFold release to download |
| `TF_SHA256` | `d17ba88f98d6e8b63141ed931c75db4561ed4ce7ba4ca7fbf8e28d98bbc27187` | sha256 of the release tarball; change only together with `TF_VERSION` |
| `THINKING` | `off` | `off` starts the server with `--no-thinking` (matches the measured rows); `on` leaves the template default |
| `NAME` | `qwen3.8-flash-next-nvfp4` | model id reported by `/v1/models` |

## Measured on one RTX PRO 6000 Blackwell (96 GB, sm_120)

TensorFold 1.2.9 release (`tensorfold-1.2.9-linux-x86_64.tar.gz`, sha256 `d17ba88f…27187`), thinking off, greedy decode.

| row | result |
|---|---|
| decode, one stream (768 new tokens) | **152 tok/s** |
| prefill, 32,797-token prompt, cold | **6,702 tok/s** |
| context | **262,144** tokens |
| needle at 130K and at 250K tokens | **found / found** |
| greedy output repeats exactly | yes |
| 4 short smoke prompts (math, fact, code, exact JSON) | 4 / 4 |

### Several agents at once

`--parallel 8 --context 32768`, prompt cache on. Each agent shares an ~8K-token system prompt and sends 3 turns with
256-token replies. Total = all new tokens / wall time. 0 errors in every row.

| agents | total tok/s | per agent | first token p50 |
|---|---:|---:|---:|
| 1 | 114 | 161 | 0.11 s |
| 4 | 315 | 92 | 0.12 s |
| 8 | **416** | 61 | 0.14 s |
| 16 (8 wait in line) | 370 | 50 | 5.6 s |

### Four agents at the full window (experimental)

`TF_FLASHNEXT_KV_PAGES=256 --parallel 4 --context 262144`: 4 agents each send a different ~241K-token document at the
same time. **4 / 4 needles found**, in 51–185 s. The streams share one page pool, so some requests wait for pages.
The prompt cache is off in this mode.

## Quality

KL divergence against the BF16 model, teacher-forced, per token, KL(BF16 ‖ quant), held-out prose and code.
The n-gram table's error is included. Harness: [tools/q38fn_kld.py](tools/q38fn_kld.py).

| check | this quant | plain round-to-nearest NVFP4 (same layout) |
|---|---|---|
| mean KL | **0.083** | 0.107 |
| top-1 token agreement | **89.3%** | 88.8% |
| perplexity (BF16 = 6.19) | **6.22** | n/a |

Calibration: WikiText-2 train articles and Python source, with zero documents shared with the evaluation set.

## How it was built

1. Banked the BF16 base model at the revision above (131 shards, 360.0 GB).
2. Plain NVFP4 export of the routed experts and requantized n-gram table in the ModelOpt `MIXED_PRECISION`
   layout — the round-to-nearest pack the GPTQ run starts from.
3. GPTQ on the NVFP4 grid: sequential per-layer calibration of the routed experts on 262,144 calibration tokens
   (WikiText-2 train + Python source; zero overlap with the evaluation set). Dense parts stay BF16.
4. Assembled the final pack: everything except the expert weight/scale tensors is byte-identical to the
   round-to-nearest pack (shard names, index, config).
5. Quality gate (KLD harness above), full-window and concurrency checks.

Inputs, commands and runtime: [tools/README.md](tools/README.md). The builders ship in [tools/](tools/);
the calibration text (WikiText-2, exllamav3 source) is not redistributed.

## Not measured yet

- Aggregate throughput of `full4` (measured only as 4/4 needles found in 51–185 s).
- Peak VRAM per mode and cold-load time.
- Any GPU other than one RTX PRO 6000 Blackwell, or any engine other than TensorFold 1.2.9 on sm_120.
- Contexts beyond 262,144: this recipe serves the native window; a longer YaRN-scaled window is not offered on CUDA.

## Licence

The Qwen Community License 1.0 applies to the weights. Its clause 2:

> If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, the licensee
> shall obtain a separate license from Qwen before Using the Software or its derivative works for any commercial
> purpose. The foregoing requirement shall not apply to the licensee's internal Use of the Software, provided that
> such Use does not make the Software, its outputs, or its underlying model capabilities available to any third party.

"Model as a Service" and "AI Work Assistant" are defined terms in that licence — read its full text (it ships with the
weights and is linked from the model card) before hosting the model for others.
The scripts and tools in this repository are Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

## Credits

- **Qwen** — base model and weights, Qwen Community License 1.0.
- **TensorFold** — Ash Hart and contributors, Apache-2.0: the engine and its NVFP4 / ModelOpt loader. Downloaded at run
  time, not redistributed.
- **GPTQ** — Frantar et al., 2022: the calibration method.
- **WikiText-2** (CC BY-SA 4.0) and **exllamav3** Python source (MIT, turboderp): calibration/evaluation text only,
  not redistributed.
