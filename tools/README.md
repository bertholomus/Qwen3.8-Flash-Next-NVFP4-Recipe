# tools — rebuild the pack, re-run the quality gate and the benches

These are the builders used to make [bertholomus/Qwen3.8-Flash-Next-NVFP4-GPTQ](https://huggingface.co/bertholomus/Qwen3.8-Flash-Next-NVFP4-GPTQ)
and its measured quality numbers (README.md). They ship as code; the calibration and evaluation **text is not
redistributed** here.

## Inputs

| input | source | licence |
|---|---|---|
| BF16 base model | Qwen/Qwen3.8-Flash-Next @ `812d96e10a0dd82b69d749e68693698cf6c36daa` (131 shards, 360.0 GB) | Qwen Community License 1.0 |
| round-to-nearest NVFP4 pack | built from the base in the ModelOpt `MIXED_PRECISION` layout: routed experts NVFP4 (E2M1, FP8 scale per 16, FP32 tensor scale), n-gram table requantized to NVFP4 rows, everything else BF16 | — |
| WikiText-2 raw train + test | any raw WikiText-2 dump with the usual ` = Title = ` article headers (the builders take a parquet with a `text` column) | CC BY-SA 4.0 |
| exllamav3 Python source | github.com/turboderp/exllamav3 | MIT (turboderp) |
| TensorFold source + docs (v1.2.6) | github.com/ashhart/TensorFold | Apache-2.0 |

The plain NVFP4 export and the n-gram table requantizer (the round-to-nearest pack the GPTQ run starts from) are
**not part of this repository**. The shipped GPTQ and KLD tools import that converter module by path for the NVFP4
grid and emit helpers (`quantize_batch`, `dequant_batch`, `_emit_proj`): put a converter at `tools/q38fn_convert.py`
or point `CONVERTER_DIR` at a directory containing it. Without it, `q38fn_gptq.py` refuses to start with a clear
message; `q38fn_kld.py` degrades to the checks that need no converter.

## What each tool does

| tool | role |
|---|---|
| `build_kld_eval.py` | builds `kld_eval.jsonl`: WikiText-2 test articles + TensorFold v1.2.6 source/docs (files over 2000 chars), round-robin interleaved, deterministic |
| `build_cal.py` | builds `cal.jsonl`: WikiText-2 **train** articles + exllamav3 Python source; asserts zero shared texts with `kld_eval.jsonl` |
| `q38fn_gptq.py` | sequential per-layer GPTQ of the routed experts on the NVFP4 grid (output-error objective, per-block scale search, error feedback through the Cholesky of the inverse Hessian); `--selftest` (CPU, minutes), `--layers N` pilot, `--resume`, `--assemble` writes the final pack — everything except expert weight/scale tensors is byte-identical to the round-to-nearest pack (shard names, index, config) |
| `q38fn_kld.py` | the quality gate: teacher-forced per-token KL(BF16 ‖ quant) vs the BF16 source, top-1 agreement, perplexity; structural self-tests with `--selftest` |
| `q38fn_conc.py` | agent-style concurrency bench against a served OpenAI-compatible endpoint: aggregate tok/s, per-agent decode tok/s, TTFT p50/p90, cached-token counts, errors |
| `build_f2.py` | **experimental, not used by this release**: converts the dense projections (attention, DeltaNet, shared experts) of a pack to MXFP8 (E4M3 codes, one UE8M0 scale byte per 32 values); kept for future experiments |

## Commands (one RTX PRO 6000; PyTorch CUDA build, safetensors, numpy, pyarrow)

```bash
# 1. evaluation set (CPU, minutes)
python3 tools/build_kld_eval.py --wt2-parquet wt2-test.parquet \
    --tensorfold-src /path/to/tensorfold-1.2.6-source --out kld_eval.jsonl

# 2. calibration set (CPU, minutes); reads the eval set to prove disjointness
python3 tools/build_cal.py --wt2-parquet wt2-train.parquet \
    --exllamav3-src /path/to/exllamav3 --kld-eval kld_eval.jsonl --out cal.jsonl

# 3. self-tests (CPU, minutes) — structural checks against the base and the RTN pack
python3 tools/q38fn_kld.py --src BASE_BF16 --pack RTN_PACK --selftest
python3 tools/q38fn_gptq.py --src BASE_BF16 --f1 RTN_PACK --cal cal.jsonl \
    --eval-ref kld_eval.jsonl --selftest

# 4. pilot: measure your per-layer rate (GPU; minutes for --layers 2)
python3 tools/q38fn_gptq.py --src BASE_BF16 --f1 RTN_PACK --cal cal.jsonl \
    --eval-ref kld_eval.jsonl --work gptq-work --layers 2

# 5. full calibration run (GPU; hours on one RTX PRO 6000 — scale the pilot's
#    per-layer time by 48 layers; resumable per layer)
python3 tools/q38fn_gptq.py --src BASE_BF16 --f1 RTN_PACK --cal cal.jsonl \
    --eval-ref kld_eval.jsonl --work gptq-work --resume

# 6. assemble the final pack (fast; the expert tensors come from --work, the rest from --f1)
python3 tools/q38fn_gptq.py --src BASE_BF16 --f1 RTN_PACK --out PACK --work gptq-work --assemble

# 7. quality gate (GPU; the numbers in README.md come from this harness)
python3 tools/q38fn_kld.py --src BASE_BF16 --pack PACK --eval kld_eval.jsonl --out kld.json

# 8. concurrency bench against a served pack (CPU client only)
python3 tools/q38fn_conc.py --url http://127.0.0.1:8080 \
    --model qwen3.8-flash-next-nvfp4 --out conc.json
```

Defaults point at relative paths (`./base-bf16`, `./pack-rtn`, `./pack`, `./cal.jsonl`, `./kld_eval.jsonl`,
`./gptq-work`) — pass everything explicitly as above. The full run is hours of GPU time on one RTX PRO 6000
(48 layers × 512 experts × 3 projections, column-sequential GPTQ plus a second pass per layer); measure your rate
with the `--layers 2` pilot instead of trusting any estimate.

## Notes

- Calibration text: WikiText-2 train (CC BY-SA 4.0) and exllamav3 Python source (MIT, turboderp) are inputs only —
  neither is redistributed by this repository.
- The evaluation set uses WikiText-2 test and TensorFold v1.2.6 source/docs; `build_cal.py` proves the two sets
  share zero documents.
- GPTQ: Frantar et al., 2022 (the calibration method).
- The on-disk pack format follows the ModelOpt `MIXED_PRECISION` NVFP4 layout (E2M1 codes, one FP8 E4M3 scale per
  16 values, one FP32 scale per tensor, `input_scale = 1.0`), W4A16: activations stay BF16.
