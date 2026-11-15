#!/usr/bin/env python3
"""q38fn_gptq.py — calibrated NVFP4 routed experts (GPTQ on the NVFP4 grid).

Sequential layer-by-layer GPTQ that keeps the round-to-nearest NVFP4 pack
format byte-compatible (TensorFold v1.2.6 modelopt-mixed NVFP4, W4A16) but chooses
codes/scales to minimise each expert's OUTPUT error on calibration activations instead
of the weight error (plain RTN: per-weight relative error 0.094-0.095, KLD kl_mean 0.1065).

Per decoder layer i (non-expert weights BF16 from the source, PLE rows dequantized
lazily from the RTN pack, exactly the q38fn_kld.run loop shape):
  pass 1: run the current hidden states through the BF16 layer with a forward hook on
          mlp.experts that accumulates, per expert e, in fp32 (plain sum, no routing
          weights — standard GPTQ): H_gu[e] = sum x_e^T x_e (2560x2560 over routed
          expert inputs), H_dn[e] = sum h_e^T h_e (640x640, h_e = act(gate)*up with the
          BF16 gate_up weights), n[e] = routed token count.
  GPTQ:   per expert, gate/up/down as SEPARATE 2-D tensors (the converter's split):
          n[e] >= --min-tokens -> GPTQ on the NVFP4 grid; else RTN fallback through
          q38fn_convert.quantize_batch (logged).
  pass 2: write the quantized experts back into the layer (dequantized bf16) and RE-RUN
          layer i on the same inputs, so layer i+1 calibrates on the output of the
          already-quantized layers 0..i.

GPTQ on the NVFP4 grid, per 2-D W [out, in] with raw H [in, in]:
  dead inputs (diag(H)==0): H_jj = 1, W[:, j] = 0 (canonical order: dead first, then
  damp); H += --damp * mean(diag(H)) * I; Hinv = cholesky(cholesky_inverse(cholesky(H)),
  upper=True). s2 = amax(W)/(6*448) exactly as the converter's rule. Columns left
  to right, act-order OFF, lazy batch updates every 128 columns. At each 16-column NVFP4
  block the scales come from the CURRENT (error-updated) weights: amax_b/6/s2 -> e4m3
  (RNE, clamp 448). --mse-scale (default on) searches the per-row scale over
  {1.0, 0.95, 0.9, 0.85, 0.8} x amax-scale minimising ||w - q(w)||^2 weighted by
  diag(Hinv^-1) = 1/Hinv_jj for the block; the stored scale is the e4m3 value. Column j
  quantizes to the nearest E2M1 value times the block scale (converter tie rule), then
  standard error feedback err = (w - q)/Hinv_jj through the Cholesky-of-inverse.
  Output layout is exactly q38fn_convert.quantize_batch's: low nibble = even column,
  weight U8 [out, in//2], weight_scale F8_E4M3 [out, in/16], weight_scale_2 F32 scalar,
  input_scale F32 1.0.

On-disk contract: everything except the decoder-layer expert weight /
weight_scale / weight_scale_2 values is byte-for-byte from the --f1 pack: MTP experts (RTN),
n-gram rows, every BF16 tensor, shard names, index, config.json.

Resume: --work holds cal_meta.json, one layer_{i:02d}.safetensors + layer_{i:02d}.done
marker per finished layer, boundary hidden states hidden_{i:02d}.safetensors (bf16; the
older boundary is deleted once a layer finishes), and stats.jsonl. --resume accepts a
saved run planned for fewer layers (a --layers N pilot that finished a prefix) and widens
the plan to the full model — finished layers and the boundary hidden states do not depend
on the planned total; every other saved parameter must match exactly. --assemble writes
the final pack from the per-layer outputs plus the RTN pack's other tensors.

CLI:
  python3 q38fn_gptq.py --src DIR --f1 DIR --out DIR --cal FILE \
      --cal-tokens 262144 --seq-len 4096 [--layers N] [--resume] [--assemble] [--selftest]

Imports by path with importlib (the way q38fn_kld.py loads the converter; never copied):
  q38fn_kld.py: st_index, st_open, st_load, build_config, build_model_parts, clone_meta,
    materialize, load_layer_bf16, patch_ple, patch_ple_hooks, build_tokens, dequant_nvfp4.
  q38fn_convert.py: quantize_batch, dequant_batch, _e2m1_code, E2M1_VALS, PROJS, _emit_proj.

Runtime: selftest minutes (CPU-capable); the full run is hours (512 experts x 3 tensors x
48 layers of column-sequential GPTQ plus two layer passes per layer).
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, math, os, re, shutil, time

import torch
import torch.nn as nn
from safetensors.torch import save_file

DEV = "cuda"
SCALE_BLOCK = 16        # NVFP4 scale block width (one e4m3 scale column)
LAZY_BLOCK = 128        # GPTQ lazy batch-update width (brief allows 128)
MSE_GRID = (1.0, 0.95, 0.9, 0.85, 0.8)
EVAL_SHA_PREFIX = "2517f5f6"   # sha256 of kld_eval.jsonl (built by build_kld_eval.py)


def fail(step, msg):
    raise RuntimeError(f"[FAIL step={step}] {msg}")


# ---------------------------------------------------------------- module loading
def _load_module(name, candidates, step):
    for p in candidates:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location(name, p)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                fail(step, f"could not import {p}: {e}")
            return mod
    fail(step, f"none of {candidates} found — put q38fn_kld.py / q38fn_convert.py next to "
               f"this file, or point CONVERTER_DIR at a directory that has them")


_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.environ.get("CONVERTER_DIR", _HERE)
KLD = _load_module("q38fn_kld",
                   [os.path.join(_HERE, "q38fn_kld.py"), os.path.join(_TOOLS_DIR, "q38fn_kld.py")],
                   "import-kld")
CONV = _load_module("q38fn_convert",
                    [os.path.join(_HERE, "q38fn_convert.py"), os.path.join(_TOOLS_DIR, "q38fn_convert.py")],
                    "import-conv")
PROJS = CONV.PROJS

_E2M1_TAB = {}


def _e2m1_tab(device):
    """E2M1 grid values (from the converter) as a fp32 lookup table, cached per device."""
    t = _E2M1_TAB.get(device)
    if t is None:
        t = torch.tensor(CONV.E2M1_VALS, dtype=torch.float32, device=device)
        _E2M1_TAB[device] = t
    return t


# ---------------------------------------------------------------- NVFP4 grid
def _quant_col(w, dec_ratio, dec, amax_b):
    """Quantize one weight column to the NVFP4 grid with the converter's exact rounding.
    w [R] fp32 current column; dec [R] fp32 block decode (stored_e4m3 * s2);
    dec_ratio [R] = dec.clamp_min(1e-12); amax_b [R] current block amax.
    Returns (code [R] uint8 with sign bit, q [R] fp32 = E2M1(code) * dec).
    Bit-identical to q38fn_convert.quantize_batch on the same weights/scales."""
    ratio = w / dec_ratio
    code = CONV._e2m1_code(ratio) | ((ratio < 0).to(torch.uint8) * 8)
    code = torch.where(amax_b > 0, code, torch.zeros_like(code))
    vals = _e2m1_tab(w.device)[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
    return code, vals * dec


def _mse_block_scale(w_blk, base, amax_b, s2, wgt):
    """--mse-scale: per-row block-scale search over MSE_GRID x amax-scale, minimising
    sum_j diag(Hinv^-1)_j * (w - q(w))^2 over the block. Candidates are e4m3-representable
    (the stored scale IS the e4m3 cast); exact ties keep the earlier grid entry (larger
    scale, m=1.0 first). w_blk [R, 16] current weights, base [R] amax/6/s2 clamped to
    [0, 448], s2 scalar tensor, wgt [16] = 1/Hinv_jj for the block's columns.
    Returns the chosen stored scale [R] as float8_e4m3fn."""
    best_bits, best_err = None, None
    for m in MSE_GRID:
        cand = (base * m).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
        dec = cand.float() * s2
        ratio = w_blk / dec.clamp_min(1e-12).unsqueeze(1)
        code = CONV._e2m1_code(ratio) | ((ratio < 0).to(torch.uint8) * 8)
        code = torch.where((amax_b > 0).unsqueeze(1), code, torch.zeros_like(code))
        vals = _e2m1_tab(w_blk.device)[(code & 7).long()] * torch.where((code & 8) > 0, -1.0, 1.0)
        q = vals * dec.unsqueeze(1)
        err = ((w_blk - q).pow(2) * wgt).sum(dim=1)
        if best_bits is None:
            best_bits, best_err = cand.view(torch.uint8), err
        else:
            take = err < best_err          # strict: ties keep the earlier (larger) scale
            best_err = torch.where(take, err, best_err)
            best_bits = torch.where(take, cand.view(torch.uint8), best_bits)
    return best_bits.view(torch.float8_e4m3fn)


def prep_hinv(H, damp):
    """Canonical GPTQ Hessian prep: dead inputs first (diag 0 -> H_jj = 1, the caller
    zeroes W's dead columns), then damp H += damp * mean(diag) * I, then the
    Cholesky-of-inverse. Returns (Hinv upper-triangular fp32, dead [in] bool)."""
    step = "prep-hinv"
    if H.dtype != torch.float32 or H.dim() != 2 or H.shape[0] != H.shape[1]:
        fail(step, f"H must be square fp32, got {tuple(H.shape)} {H.dtype}")
    Hd = H.detach().clone()
    dead = torch.diagonal(Hd) == 0
    if bool(dead.any()):
        Hd[dead, dead] = 1.0
    damp_val = float(damp) * float(torch.diagonal(Hd).mean())
    if not damp_val > 0.0:
        fail(step, f"non-positive damping value {damp_val} (H all zero — call RTN instead)")
    Hd.diagonal().add_(damp_val)
    L = torch.linalg.cholesky(Hd)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    return Hinv, dead


def gptq_nvfp4(W, H, damp=0.01, mse=True, s2=None, Hinv=None, dead=None):
    """GPTQ on the NVFP4 grid for one 2-D tensor.
    W fp32 [out, in] — the ORIGINAL tensor (never modified). H fp32 [in, in] — the RAW
    (undamped) covariance; only used when Hinv is not supplied.
    s2: per-tensor global scale, default amax(W)/(6*448) — the converter's rule.
    Hinv/dead: pass both to reuse across tensors sharing H (gate/up share H_gu).
    Returns (packed U8 [out, in//2], stored F8_E4M3 [out, in//16], s2 F32 0-dim,
             Q fp32 [out, in] — the dequantized tensor GPTQ used; its bf16 cast is what
             goes into the layer and what q38fn_kld.dequant_nvfp4 decodes the codes to)."""
    step = "gptq"
    if W.dim() != 2 or W.dtype != torch.float32:
        fail(step, f"W must be 2-D fp32, got {tuple(W.shape)} {W.dtype}")
    R, C = W.shape
    if C % SCALE_BLOCK != 0:
        fail(step, f"in dim {C} not divisible by {SCALE_BLOCK}")
    if Hinv is None:
        if H is None:
            fail(step, "need H when Hinv is not given")
        if tuple(H.shape) != (C, C):
            fail(step, f"H must be [{C},{C}], got {tuple(H.shape)}")
        Hinv, dead = prep_hinv(H, damp)
    elif dead is None:
        fail(step, "Hinv reuse requires the matching dead mask")
    if s2 is None:
        amax_all = W.abs().reshape(-1, SCALE_BLOCK).amax(dim=-1).max()
        s2 = (amax_all / (6.0 * 448.0)).float().reshape(())
    s2c = s2.clamp(min=1e-30)
    Ww = W.clone()
    if dead is not None and bool(dead.any()):
        Ww[:, dead] = 0.0
    dev = W.device
    codes = torch.zeros(R, C, dtype=torch.uint8, device=dev)
    scales = torch.zeros(R, C // SCALE_BLOCK, dtype=torch.float8_e4m3fn, device=dev)
    hinv_diag = torch.diagonal(Hinv)
    for i0 in range(0, C, LAZY_BLOCK):                # lazy batch-update blocks
        i2 = min(i0 + LAZY_BLOCK, C)
        W1 = Ww[:, i0:i2].clone()                     # working copy for this lazy block
        Hinv1 = Hinv[i0:i2, i0:i2]
        err_acc = torch.zeros(R, i2 - i0, dtype=torch.float32, device=dev)
        for b0 in range(i0, i2, SCALE_BLOCK):         # NVFP4 scale blocks
            b2 = min(b0 + SCALE_BLOCK, C)
            w_blk = W1[:, b0 - i0:b2 - i0]            # CURRENT (error-updated) weights
            amax_b = w_blk.abs().amax(dim=1)
            base = (amax_b / 6.0 / s2c).clamp(0.0, 448.0)
            if mse:
                stored = _mse_block_scale(w_blk, base, amax_b, s2, 1.0 / hinv_diag[b0:b2])
            else:
                stored = base.to(torch.float8_e4m3fn)
            scales[:, b0 // SCALE_BLOCK] = stored
            dec = stored.float() * s2
            dec_ratio = dec.clamp_min(1e-12)
            for j in range(b0, b2):                   # column-sequential GPTQ
                ii = j - i0
                code, q = _quant_col(W1[:, ii], dec_ratio, dec, amax_b)
                codes[:, j] = code
                err = (W1[:, ii] - q) / hinv_diag[j]
                err_acc[:, ii] = err
                if ii + 1 < i2 - i0:                  # within-lazy-block feedback
                    W1[:, ii + 1:].addcmul_(err.unsqueeze(1),
                                             Hinv1[ii, ii + 1:].unsqueeze(0), value=-1.0)
        if i2 < C:                                    # deferred update to the rest
            Ww[:, i2:] -= err_acc @ Hinv[i0:i2, i2:]
    # dequantized tensor GPTQ used — arithmetic identical to q38fn_kld.dequant_nvfp4
    vals = _e2m1_tab(dev)[(codes & 7).long()] * torch.where((codes & 8) > 0, -1.0, 1.0)
    dec_full = (scales.float() * s2).repeat_interleave(SCALE_BLOCK, dim=1)
    Q = vals * dec_full
    c = codes.reshape(R, C // SCALE_BLOCK, SCALE_BLOCK // 2, 2)
    packed = (c[..., 0] | (c[..., 1] << 4)).reshape(R, C // 2).contiguous()
    return packed, scales.contiguous(), s2, Q


def rtn_nvfp4(W):
    """RTN fallback through the converter (quantize_batch) on one 2-D tensor [R, C] fp32.
    Returns (packed, stored, s2, Q) with the same contract as gptq_nvfp4."""
    R, C = W.shape
    packed, stored, s2 = CONV.quantize_batch(W.reshape(1, R, C).contiguous())
    Q = CONV.dequant_batch(packed, stored, s2, C)[0]
    return packed[0], stored[0], s2[0], Q


def emit_expert(out, P, e, proj, packed, stored, s2):
    """Per-expert tensors in exactly the converter's emit layout
    (q38fn_convert._emit_proj): weight U8 [R, C//2], weight_scale F8_E4M3 [R, C//16],
    weight_scale_2 F32 scalar, input_scale F32 1.0 (metadata only, W4A16)."""
    base = f"{P}.experts.{e}.{proj}"
    out[base + ".weight"] = packed.detach().cpu().clone()
    out[base + ".weight_scale"] = stored.detach().cpu().clone()
    out[base + ".weight_scale_2"] = s2.detach().cpu().reshape(())
    out[base + ".input_scale"] = torch.ones((), dtype=torch.float32)


# ---------------------------------------------------------------- expert H accumulation
class HAccumulator:
    """Forward hook on mlp.experts: per-expert covariances in fp32, plain sum over the
    calibration tokens routed to each expert (brief §2a — no routing-weight weighting)."""
    def __init__(self, n_experts, hidden, inter, device):
        self.H_gu = torch.zeros(n_experts, hidden, hidden, dtype=torch.float32, device=device)
        self.H_dn = torch.zeros(n_experts, inter, inter, dtype=torch.float32, device=device)
        self.n = torch.zeros(n_experts, dtype=torch.long, device=device)

    def hook(self, module, args, output):
        step = "h-hook"
        if len(args) < 2:
            fail(step, f"mlp.experts forward got {len(args)} positional args, expected 3")
        x, tk = args[0], args[1]   # x [ntok, hidden] bf16 expert inputs; tk [ntok, top_k]
        if x.dim() != 2:
            fail(step, f"expert input must be 2-D, got {tuple(x.shape)}")
        gu_w = module.gate_up_proj   # [E, 2*inter, hidden] bf16 (BF16 source at pass-1 time)
        act = module.act_fn
        for e in range(module.num_experts):
            tok = (tk == e).any(dim=-1).nonzero().squeeze(-1)
            if tok.numel() == 0:
                continue
            x_e = x.index_select(0, tok)
            xf = x_e.float()
            self.H_gu[e] += xf.T @ xf
            gate, up = nn.functional.linear(x_e, gu_w[e]).chunk(2, dim=-1)
            h_e = (act(gate) * up).float()     # act(gate)*up with the BF16 gate_up weights
            self.H_dn[e] += h_e.T @ h_e
            self.n[e] += tok.numel()


def rel_out_err(W, Q, H_raw):
    """||X W^T - X Q^T||_F / ||X W^T||_F from the raw (undamped) H = X^T X —
    the relative OUTPUT error on the calibration inputs, without keeping X."""
    D = W - Q
    num = float(((D @ H_raw) * D).sum())
    den = float(((W @ H_raw) * W).sum())
    if den <= 0.0:
        return 0.0
    return math.sqrt(max(num, 0.0) / den)


def quantize_experts_layer(layer, acc, args, P, layer_idx):
    """GPTQ (or RTN fallback) every expert of one materialized layer, write the
    dequantized bf16 experts back into the layer (for pass 2), and return
    (layer-file tensor dict, stats). Fails loudly when GPTQ does not beat RTN on a
    sampled tensor (brief §4)."""
    ex = layer.mlp.experts
    E, inter = ex.num_experts, ex.intermediate_dim
    gu, dn = ex.gate_up_proj.data, ex.down_proj.data
    mse = args.mse_scale == "on"
    n_cpu = acc.n.detach().cpu().tolist()
    eligible = [e for e in range(E) if n_cpu[e] >= args.min_tokens]
    rtn_experts = [e for e in range(E) if n_cpu[e] < args.min_tokens]
    sampled = sorted(eligible, key=lambda e: (-n_cpu[e], e))[:4]   # most-routed, deterministic
    sampled_set = set(sampled)
    out, samples = {}, []
    for e in range(E):
        Wgu = gu[e].float()
        Wg, Wu = Wgu[0:inter], Wgu[inter:2 * inter]
        Wd = dn[e].float()
        if n_cpu[e] >= args.min_tokens:
            Hinv_gu, dead_gu = prep_hinv(acc.H_gu[e], args.damp)
            rg = gptq_nvfp4(Wg, acc.H_gu[e], args.damp, mse, Hinv=Hinv_gu, dead=dead_gu)
            ru = gptq_nvfp4(Wu, acc.H_gu[e], args.damp, mse, Hinv=Hinv_gu, dead=dead_gu)
            del Hinv_gu
            Hinv_dn, dead_dn = prep_hinv(acc.H_dn[e], args.damp)
            rd = gptq_nvfp4(Wd, acc.H_dn[e], args.damp, mse, Hinv=Hinv_dn, dead=dead_dn)
            del Hinv_dn
        else:
            rg, ru, rd = rtn_nvfp4(Wg), rtn_nvfp4(Wu), rtn_nvfp4(Wd)
        gu[e, 0:inter] = rg[3].to(torch.bfloat16)
        gu[e, inter:2 * inter] = ru[3].to(torch.bfloat16)
        dn[e] = rd[3].to(torch.bfloat16)
        emit_expert(out, P, e, "gate_proj", rg[0], rg[1], rg[2])
        emit_expert(out, P, e, "up_proj", ru[0], ru[1], ru[2])
        emit_expert(out, P, e, "down_proj", rd[0], rd[1], rd[2])
        if e in sampled_set:
            for proj, W2d, res, Hraw in (("gate_proj", Wg, rg, acc.H_gu[e]),
                                         ("up_proj", Wu, ru, acc.H_gu[e]),
                                         ("down_proj", Wd, rd, acc.H_dn[e])):
                e_rtn = rel_out_err(W2d, rtn_nvfp4(W2d)[3], Hraw)
                e_gptq = rel_out_err(W2d, res[3], Hraw)
                if not e_gptq < e_rtn:
                    fail("stats", f"layer {layer_idx} expert {e} {proj}: GPTQ rel output err "
                         f"{e_gptq:.6f} NOT lower than RTN {e_rtn:.6f} (n={n_cpu[e]})")
                samples.append({"expert": e, "proj": proj, "n": n_cpu[e],
                                "rel_err_rtn": round(e_rtn, 6),
                                "rel_err_gptq": round(e_gptq, 6)})
    if rtn_experts:
        print(f"  layer {layer_idx}: RTN fallback for {len(rtn_experts)} experts "
              f"(n < {args.min_tokens}): {rtn_experts[:12]}"
              f"{'...' if len(rtn_experts) > 12 else ''}", flush=True)
    stats = {"experts_gptq": len(eligible), "experts_rtn": len(rtn_experts),
             "rtn_experts": rtn_experts[:64], "sampled": samples}
    return out, stats


# ---------------------------------------------------------------- calibration data
def _doc_hashes(path, step):
    hs = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            if "text" not in d:
                fail(step, f"{path}: jsonl line without a 'text' field")
            hs.append(hashlib.sha256(d["text"].encode()).hexdigest())
    return hs


def check_cal_disjoint(args):
    """Brief §1: the calibration set must share zero documents with the KLD eval set."""
    step = "cal-disjoint"
    if not os.path.isfile(args.eval_ref):
        fail(step, f"eval set {args.eval_ref} missing — cannot verify disjointness")
    esha = hashlib.sha256(open(args.eval_ref, "rb").read()).hexdigest()
    if not esha.startswith(EVAL_SHA_PREFIX):
        fail(step, f"{args.eval_ref} sha256 {esha[:16]}... does not start with "
                   f"{EVAL_SHA_PREFIX} (wrong eval file — rebuild kld_eval.jsonl with build_kld_eval.py)")
    if not os.path.isfile(args.cal):
        fail(step, f"calibration file {args.cal} missing")
    cal_sha = hashlib.sha256(open(args.cal, "rb").read()).hexdigest()
    ev, cal = _doc_hashes(args.eval_ref, step), _doc_hashes(args.cal, step)
    overlap = sorted(set(cal) & set(ev))
    if overlap:
        fail(step, f"{len(overlap)} calibration documents also appear in the KLD eval set "
                   f"(e.g. sha256 {overlap[0][:16]}...) — pick a disjoint --cal file")
    return {"cal_docs": len(cal), "cal_file_sha256": cal_sha, "eval_sha256": esha, "overlap": 0}


def save_boundary(work, name, h):
    save_file({"hidden": h.detach().to("cpu").contiguous()},
              os.path.join(work, name + ".safetensors"))


def load_boundary(work, name, shape, dev):
    from safetensors import safe_open
    p = os.path.join(work, name + ".safetensors")
    if not os.path.isfile(p):
        fail("resume", f"boundary hidden states {p} missing — re-run without --resume")
    with safe_open(p, framework="pt", device="cpu") as f:
        t = f.get_tensor("hidden")
    if tuple(t.shape) != shape or t.dtype != torch.bfloat16:
        fail("resume", f"boundary {p}: {tuple(t.shape)} {t.dtype}, expected {shape} bf16")
    return t.to(dev)


def remove_boundary(work, name):
    p = os.path.join(work, name + ".safetensors")
    if os.path.isfile(p):
        os.remove(p)


RESUME_KEYS = ("src", "f1", "cal", "cal_file_sha256", "cal_ids_sha256", "cal_tokens",
               "seq_len", "nseq", "mb", "damp", "min_tokens", "mse_scale",
               "layers_planned", "eval_sha256")


def resume_guard(old, meta, n_layers):
    """Resume policy: every RESUME_KEYS entry must match the saved run exactly, except
    layers_planned, which may only WIDEN. The intended flow is a --layers N pilot that
    finishes a prefix, then --resume for the full model: finished layers and the
    boundary hidden states do not depend on the planned total, so 2 -> 48 is safe,
    while a shrink (48 -> 2) would silently truncate --assemble. Returns the saved
    layers_planned; raises FAIL(resume) on any other difference."""
    mism = {k: [old.get(k), meta[k]] for k in RESUME_KEYS
            if old.get(k) != meta[k] and k != "layers_planned"}
    if mism:
        fail("resume", f"calibration/params changed since the saved run: {json.dumps(mism)[:500]}")
    old_plan = old.get("layers_planned")
    if not isinstance(old_plan, int) or old_plan < 1:
        fail("resume", f"saved run has no valid layers_planned ({old_plan!r})")
    if old_plan > n_layers:
        fail("resume", f"saved run planned {old_plan} layers, this invocation plans "
                       f"{n_layers} — the plan may widen (pilot -> full), not shrink")
    return old_plan


@torch.no_grad()
def run(args):
    step = "run"
    if not torch.cuda.is_available():
        fail(step, "the calibration run needs CUDA (RTX PRO 6000, sm_120); "
                   "--selftest is the CPU-capable path")
    t00 = time.time()
    os.makedirs(args.work, exist_ok=True)
    cal_info = check_cal_disjoint(args)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src)
    cal_docs = [json.loads(l) for l in open(args.cal) if l.strip()]
    if args.cal_tokens % args.seq_len != 0:
        fail(step, f"--cal-tokens {args.cal_tokens} is not a multiple of --seq-len {args.seq_len}")
    nseq = args.cal_tokens // args.seq_len
    ids = KLD.build_tokens(tok, cal_docs, args.seq_len, nseq)   # deterministic order
    cal_ids_sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    ids = ids.to(DEV)
    print(f"cal: {cal_info['cal_docs']} docs -> {nseq} x {args.seq_len} tokens, "
          f"token-ids sha256 {cal_ids_sha}, zero overlap with {args.eval_ref}", flush=True)

    cfg = KLD.build_config(args.src)
    parts, M = KLD.build_model_parts(args.src, cfg)
    src_index = KLD.st_index(args.src)
    f1_index = KLD.st_index(args.f1)
    n_layers = args.layers if args.layers else len(parts["layers"])
    if n_layers > len(parts["layers"]):
        fail(step, f"--layers {n_layers} > decoder layers {len(parts['layers'])}")
    B, T, mb = nseq, args.seq_len, args.mb
    width = cfg.hc_count * cfg.hidden_size
    meta = {"src": os.path.abspath(args.src), "f1": os.path.abspath(args.f1),
            "cal": os.path.abspath(args.cal), "cal_file_sha256": cal_info["cal_file_sha256"],
            "cal_ids_sha256": cal_ids_sha, "cal_tokens": args.cal_tokens,
            "seq_len": args.seq_len, "nseq": nseq, "mb": args.mb, "damp": args.damp,
            "min_tokens": args.min_tokens, "mse_scale": args.mse_scale,
            "layers_planned": n_layers, "eval_ref": os.path.abspath(args.eval_ref),
            "eval_sha256": cal_info["eval_sha256"], "hidden_shape": [B, T, width]}
    meta_path = os.path.join(args.work, "cal_meta.json")

    def marker(i):
        return os.path.isfile(os.path.join(args.work, f"layer_{i:02d}.done"))

    def lfile(i):
        return os.path.isfile(os.path.join(args.work, f"layer_{i:02d}.safetensors"))

    touched = [i for i in range(n_layers) if marker(i) or lfile(i)]
    if args.resume:
        if not os.path.isfile(meta_path):
            fail("resume", f"no {meta_path} to resume from")
        old = json.load(open(meta_path))
        old_plan = resume_guard(old, meta, n_layers)
        if old_plan < n_layers:
            # pilot -> full: widen the saved plan so --assemble substitutes every
            # finished layer (assemble reads layers_planned from cal_meta.json)
            json.dump(meta, open(meta_path, "w"), indent=1)
            print(f"resume: plan widened {old_plan} -> {n_layers} layers "
                  f"(cal_meta.json updated)", flush=True)
        broken = [i for i in touched if not (marker(i) and lfile(i))]
        if broken:
            fail("resume", f"layers with a marker but no output file (or vice versa): {broken[:8]}")
        if touched and touched != list(range(len(touched))):
            fail("resume", f"finished layers are not a contiguous prefix: {touched[:8]}...")
        start = touched[-1] + 1 if touched else 0
        print(f"resume: layers 0..{start - 1} done, continuing at {start}", flush=True)
    else:
        if touched:
            fail(step, f"{args.work} already holds finished layers {touched[:4]}... — "
                       f"pass --resume or clean it first")
        json.dump(meta, open(meta_path, "w"), indent=1)
        start = 0

    h = torch.zeros(B, T, width, dtype=torch.bfloat16, device=DEV)
    if start == 0:
        emb_mod = KLD.materialize(KLD.clone_meta(parts["embed"]))
        KLD.load_layer_bf16(emb_mod, args.src, src_index, "model.embed_tokens")
        for s in range(0, B, mb):
            h[s:s + mb] = emb_mod(ids[s:s + mb]).to(torch.bfloat16).repeat(1, 1, cfg.hc_count)
        del emb_mod
        torch.cuda.empty_cache()
        save_boundary(args.work, "hidden_emb", h)
    else:
        h = load_boundary(args.work, f"hidden_{start - 1:02d}", (B, T, width), DEV)
        for j in range(start - 1):                    # drop stale boundaries
            remove_boundary(args.work, f"hidden_{j:02d}")
        remove_boundary(args.work, "hidden_emb")

    rotary = parts["M"].Qwen4ExpTextRotaryEmbedding(cfg).to(device=DEV)
    position_embeddings = rotary(torch.zeros(B, 1, cfg.hidden_size, dtype=torch.bfloat16, device=DEV),
                                 torch.arange(T, device=DEV).unsqueeze(0))
    position_embeddings = tuple(t.expand(B, *t.shape[1:]).contiguous() for t in position_embeddings)
    pos = torch.arange(T, device=DEV).unsqueeze(0).expand(B, -1)
    stats_path = os.path.join(args.work, "stats.jsonl")

    for i in range(start, n_layers):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        prefix = f"model.layers.{i}"
        meta_layer = parts["layers"][i]
        KLD.patch_ple(meta_layer, args.f1, f1_index, prefix, args.src, src_index)   # shrink meta PLE embed
        layer = KLD.materialize(KLD.clone_meta(meta_layer))
        KLD.patch_ple_hooks(layer, args.f1, f1_index, prefix, args.src, src_index)  # lazy RTN-pack ngram dequant
        KLD.load_layer_bf16(layer, args.src, src_index, prefix)
        cfg._attn_implementation = "sdpa"
        # ---- pass 1: BF16 layer, accumulate per-expert H (outputs discarded)
        acc = HAccumulator(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size, DEV)
        handle = layer.mlp.experts.register_forward_hook(acc.hook)
        for s in range(0, B, mb):
            pos_mb = pos[s:s + mb]
            cm = M.create_causal_mask(config=cfg, inputs_embeds=h[s:s + mb], attention_mask=None,
                                      past_key_values=None, position_ids=pos_mb,
                                      allow_is_causal_skip=False)
            layer(h[s:s + mb], position_ids=pos_mb,
                  position_embeddings=tuple(t[s:s + mb] for t in position_embeddings),
                  ple_input_ids=ids[s:s + mb], attention_mask=cm)
        handle.remove()
        # ---- GPTQ (or RTN fallback) + write the quantized experts into the layer
        P = f"model.language_model.layers.{i}.mlp"
        out, stats = quantize_experts_layer(layer, acc, args, P, i)
        del acc
        torch.cuda.empty_cache()
        # ---- pass 2: RE-RUN layer i on the same inputs with the quantized experts
        for s in range(0, B, mb):
            pos_mb = pos[s:s + mb]
            cm = M.create_causal_mask(config=cfg, inputs_embeds=h[s:s + mb], attention_mask=None,
                                      past_key_values=None, position_ids=pos_mb,
                                      allow_is_causal_skip=False)
            o = layer(h[s:s + mb], position_ids=pos_mb,
                      position_embeddings=tuple(t[s:s + mb] for t in position_embeddings),
                      ple_input_ids=ids[s:s + mb], attention_mask=cm)
            h[s:s + mb] = o[0] if isinstance(o, tuple) else o
        # ---- persist: boundary hidden states, layer output, marker; drop the older boundary
        save_boundary(args.work, f"hidden_{i:02d}", h)
        save_file(out, os.path.join(args.work, f"layer_{i:02d}.safetensors"))
        del out
        stats["layer"] = i
        stats["seconds"] = round(time.time() - t0, 1)
        stats["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        json.dump({"layer": i, "seconds": stats["seconds"],
                   "experts_gptq": stats["experts_gptq"], "experts_rtn": stats["experts_rtn"],
                   "gpu_peak_gib": stats["gpu_peak_gib"], "cal_ids_sha256": cal_ids_sha},
                  open(os.path.join(args.work, f"layer_{i:02d}.done"), "w"))
        remove_boundary(args.work, "hidden_emb" if i == 0 else f"hidden_{i - 1:02d}")
        with open(stats_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        print(f"layer {i}/{n_layers} gptq={stats['experts_gptq']} rtn={stats['experts_rtn']} "
              f"sec={stats['seconds']} peak={stats['gpu_peak_gib']}GiB", flush=True)
        del layer
        torch.cuda.empty_cache()
    print(json.dumps({"done": n_layers, "seconds": round(time.time() - t00, 1),
                      "work": args.work, "cal_ids_sha256": cal_ids_sha}))
    print(f"next: python3 q38fn_gptq.py --assemble --work {args.work} --f1 {args.f1} --out {args.out}")


# ---------------------------------------------------------------- assemble
_DEC_KEY = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2)$")
_DT_TAG = {torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3", torch.float32: "F32"}


def assemble(args):
    """Write the final pack from the per-layer outputs plus the RTN pack's other
    tensors. Same shard names, same index, same config.json; byte-identical to the RTN pack
    except the substituted decoder-layer expert weight / weight_scale / weight_scale_2
    (input_scale and every non-expert tensor, incl. MTP and the n-gram rows, stay the RTN pack's)."""
    step = "assemble"
    from safetensors import safe_open
    meta_path = os.path.join(args.work, "cal_meta.json")
    if not os.path.isfile(meta_path):
        fail(step, f"no {meta_path} — run the calibration first")
    planned = int(json.load(open(meta_path))["layers_planned"])
    f1_index = KLD.st_index(args.f1)
    out_dir = os.path.abspath(args.out)
    if out_dir == os.path.abspath(args.f1):
        fail(step, "--out must differ from --f1")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        fail(step, f"{out_dir} exists and is not empty — refusing to overwrite")
    os.makedirs(out_dir, exist_ok=True)
    expected = 0
    for i in range(planned):
        p = os.path.join(args.work, f"layer_{i:02d}.safetensors")
        mk = os.path.join(args.work, f"layer_{i:02d}.done")   # run() writes layer_XX.done markers
        if not (os.path.isfile(p) and os.path.isfile(mk)):
            fail(step, f"layer {i} not finished (need {p} + {mk})")
        with safe_open(p, framework="pt", device="cpu") as f:
            n_sub = sum(1 for k in f.keys() if _DEC_KEY.match(k))
        if n_sub == 0 or n_sub % 9 != 0:
            fail(step, f"{p}: {n_sub} substitutable expert tensors "
                       f"(expected a positive multiple of 9 = experts x 3 proj x 3 tensors)")
        expected += n_sub
    if planned < 48:
        print(f"WARNING: only {planned} layers planned; decoder layers {planned}..47 keep "
              f"RTN pack experts", flush=True)
    shards = sorted(set(f1_index.values()))
    subst_total = 0
    for shard in shards:
        src_shard = os.path.join(args.f1, shard)
        with safe_open(src_shard, framework="pt", device="cpu") as f:
            shard_keys = list(f.keys())
        needed = {}
        for k in shard_keys:
            m = _DEC_KEY.match(k)
            if m and int(m.group(1)) < planned:
                needed[k] = int(m.group(1))
        out_path = os.path.join(out_dir, shard)
        if not needed:
            os.link(src_shard, out_path)           # hardlink: byte-identical, no extra disk (the RTN pack is read-only)
            continue
        by_layer = {}
        for k, li in needed.items():
            by_layer.setdefault(li, []).append(k)
        sub = {}
        for li, keys in sorted(by_layer.items()):
            with safe_open(os.path.join(args.work, f"layer_{li:02d}.safetensors"),
                           framework="pt", device="cpu") as lf:
                for k in keys:
                    sub[k] = lf.get_tensor(k)
        out = {}
        with safe_open(src_shard, framework="pt", device="cpu") as f:
            for k in shard_keys:                   # keep the RTN pack's shard key order
                t = sub.get(k)
                if t is None:
                    out[k] = f.get_tensor(k)
                    continue
                sl = f.get_slice(k)
                tag = _DT_TAG.get(t.dtype)
                if tag is None or sl.get_dtype() != tag or tuple(sl.get_shape()) != tuple(t.shape):
                    fail(step, f"{k}: ours {tuple(t.shape)} {t.dtype} vs "
                               f"RTN pack {tuple(sl.get_shape())} {sl.get_dtype()}")
                out[k] = t
        save_file(out, out_path, metadata={"format": "pt"})
        subst_total += len(sub)
        del sub, out
        print(f"[{shard}] rewritten with {len(needed)} substituted expert tensors", flush=True)
    if subst_total != expected:
        fail(step, f"substituted {subst_total} expert tensors, expected {expected}")
    # everything else byte-for-byte: index, config.json, tokenizer, manifests, ngram bookkeeping
    for fn in sorted(os.listdir(args.f1)):
        sp = os.path.join(args.f1, fn)
        if fn.endswith(".safetensors") or fn.endswith(".safetensors.done"):
            continue
        dp = os.path.join(out_dir, fn)
        if os.path.isdir(sp):
            shutil.copytree(sp, dp, dirs_exist_ok=True)
        else:
            shutil.copy2(sp, dp)
    print(json.dumps({"out": out_dir, "layers_substituted": planned,
                      "expert_tensors_substituted": subst_total, "shards": len(shards)}))


# ---------------------------------------------------------------- selftest
@torch.no_grad()
def selftest(args):
    """Brief §selftest, must run < 10 min without the full model:
    (a) random W [640, 2560] and X [4096, 2560] with outlier columns: GPTQ output error
        < RTN output error, and the codes decode via q38fn_kld.dequant_nvfp4 to exactly
        the tensor GPTQ used;
    (b) H = I (no correlation) -> GPTQ codes == RTN codes with --mse-scale off;
    (c) layout round-trip vs q38fn_convert.quantize_batch on one real layer-0 expert.
    Prints SELFTEST PASS only if all pass. Runs on CUDA when available, else CPU."""
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"selftest on {dev} (torch {torch.__version__}) ...", flush=True)
    mse = args.mse_scale == "on"
    R, C, N = 640, 2560, 4096

    print("selftest (a) GPTQ output error < RTN on correlated X with outlier columns ...", flush=True)
    g = torch.Generator(device=dev).manual_seed(0)
    Z = torch.randn(N, 512, generator=g, device=dev, dtype=torch.float32)
    Bm = torch.randn(512, C, generator=g, device=dev, dtype=torch.float32) / math.sqrt(512.0)
    X = Z @ Bm                       # rank-512: strongly correlated columns
    X[:, 0:4] *= 30.0                # activation outlier channels
    W = torch.randn(R, C, generator=g, device=dev, dtype=torch.float32) * 0.05
    W[:, 0:16] *= 60.0               # outlier weight block (one NVFP4 block)
    W[:, 512:528] *= 25.0            # second outlier block
    H = X.T @ X
    del Z, Bm
    pk_r, st_r, s2_r = CONV.quantize_batch(W.reshape(1, R, C).contiguous())
    Qr = CONV.dequant_batch(pk_r, st_r, s2_r, C)[0]
    pk, st, s2, Q = gptq_nvfp4(W, H, damp=args.damp, mse=mse)
    if not torch.equal(KLD.dequant_nvfp4(pk, st, s2, (R, C)), Q.to(torch.bfloat16)):
        fail("selftest-a", "codes do not decode via q38fn_kld.dequant_nvfp4 to the tensor GPTQ used")
    den = float((X @ W.T).norm())
    e_rtn = float((X @ (W - Qr).T).norm()) / den
    e_gptq = float((X @ (W - Q).T).norm()) / den
    print(f"  RTN rel output err {e_rtn:.6f} vs GPTQ {e_gptq:.6f}", flush=True)
    if not e_gptq < e_rtn:
        fail("selftest-a", f"GPTQ {e_gptq:.6f} NOT < RTN {e_rtn:.6f}")
    print("  (a) OK", flush=True)

    print("selftest (b) H = I -> GPTQ codes == RTN codes (--mse-scale off) ...", flush=True)
    g = torch.Generator(device=dev).manual_seed(1)
    Wb = torch.randn(R, C, generator=g, device=dev, dtype=torch.float32)
    Heye = torch.eye(C, device=dev, dtype=torch.float32)
    Hinv_eye, dead_eye = prep_hinv(Heye, args.damp)
    pk, st, s2, _ = gptq_nvfp4(Wb, Heye, damp=args.damp, mse=False, Hinv=Hinv_eye, dead=dead_eye)
    pk_r, st_r, s2_r = CONV.quantize_batch(Wb.reshape(1, R, C).contiguous())
    if not torch.equal(pk, pk_r[0]):
        nbad = int((pk != pk_r[0]).sum())
        fail("selftest-b", f"{nbad} packed bytes differ from RTN with H=I and --mse-scale off")
    if not torch.equal(st.view(torch.uint8), st_r.view(torch.uint8)[0]):
        fail("selftest-b", "block scales differ from RTN with H=I and --mse-scale off")
    if int(s2.view(torch.int32).item()) != int(s2_r[0].view(torch.int32).item()):
        fail("selftest-b", f"s2 differs from RTN: {s2.item()} vs {s2_r[0].item()}")
    print("  (b) OK (packed codes, e4m3 scales and s2 bit-identical to quantize_batch)", flush=True)

    print("selftest (c) layout round-trip vs quantize_batch on layer 0 expert 0 ...", flush=True)
    src_index = KLD.st_index(args.src)
    gu_name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    dn_name = "model.language_model.layers.0.mlp.experts.down_proj"
    for n in (gu_name, dn_name):
        if n not in src_index:
            fail("selftest-c", f"{n} missing from the {args.src} index")
    with KLD.st_open(args.src, src_index[gu_name]) as f:
        gu = f.get_slice(gu_name)[0:1]            # [1, 1280, 2560] bf16
    with KLD.st_open(args.src, src_index[dn_name]) as f:
        dn = f.get_slice(dn_name)[0:1]            # [1, 2560, 640] bf16
    if tuple(gu.shape) != (1, 1280, 2560) or tuple(dn.shape) != (1, 2560, 640):
        fail("selftest-c", f"unexpected source shapes {tuple(gu.shape)} / {tuple(dn.shape)}")
    P = "model.language_model.layers.0.mlp"
    tensors = {"gate_proj": gu[0, 0:640].float().to(dev),
               "up_proj": gu[0, 640:1280].float().to(dev),
               "down_proj": dn[0].float().to(dev)}
    shapes = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
    eye_inv = {C: (Hinv_eye, dead_eye)}           # reuse (b)'s Cholesky for gate/up
    for proj, Wt in tensors.items():
        R2, C2 = shapes[proj]
        if tuple(Wt.shape) != (R2, C2):
            fail("selftest-c", f"{proj}: {tuple(Wt.shape)} != {(R2, C2)}")
        if C2 not in eye_inv:
            eye_inv[C2] = prep_hinv(torch.eye(C2, device=dev, dtype=torch.float32), args.damp)
        Hi, de = eye_inv[C2]
        pk, st, s2, _ = gptq_nvfp4(Wt, None, damp=args.damp, mse=False, Hinv=Hi, dead=de)
        pk_r, st_r, s2_r = CONV.quantize_batch(Wt.reshape(1, R2, C2).contiguous())
        if not (torch.equal(pk, pk_r[0])
                and torch.equal(st.view(torch.uint8), st_r.view(torch.uint8)[0])
                and int(s2.view(torch.int32).item()) == int(s2_r[0].view(torch.int32).item())):
            fail("selftest-c", f"{proj}: GPTQ(H=I, mse off) not bit-identical to "
                               f"quantize_batch on the real tensor")
        Dk = KLD.dequant_nvfp4(pk, st, s2, (R2, C2))
        Dc = CONV.dequant_batch(pk_r, st_r, s2_r, C2)[0].to(torch.bfloat16)
        if not torch.equal(Dk, Dc):
            fail("selftest-c", f"{proj}: dequant_nvfp4 vs converter dequant_batch mismatch")
        ours, ref = {}, {}
        emit_expert(ours, P, 0, proj, pk, st, s2)
        CONV._emit_proj(P, proj, pk.reshape(1, R2, C2 // 2), st.reshape(1, R2, C2 // 16),
                        s2.reshape(1), ref)
        if set(ours) != set(ref):
            fail("selftest-c", f"{proj}: emit keys {sorted(set(ours) ^ set(ref))} "
                               f"differ from converter _emit_proj")
        for k in ref:
            if not torch.equal(ours[k], ref[k]):
                fail("selftest-c", f"{proj}: emit tensor {k} differs from converter _emit_proj")
        print(f"  {proj}: bit-identical to quantize_batch; dequant round-trip + "
              f"emit layout OK", flush=True)
    print(f"SELFTEST PASS ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(
        description="Calibrated NVFP4 experts (GPTQ on the NVFP4 grid)")
    ap.add_argument("--src", default="./base-bf16",
                    help="BF16 source (read-only)")
    ap.add_argument("--f1", default="./pack-rtn", help="round-to-nearest NVFP4 pack (read-only)")
    ap.add_argument("--out", default="./pack", help="output pack (--assemble)")
    ap.add_argument("--cal", default="./cal.jsonl", help='calibration jsonl ({"text": ...})')
    ap.add_argument("--eval-ref", default="./kld_eval.jsonl",
                    help="KLD eval set the calibration must not overlap")
    ap.add_argument("--work", default="./gptq-work", help="resume/stats work dir")
    ap.add_argument("--cal-tokens", type=int, default=262144, help="calibration token budget")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=0, help="limit decoder layers (0 = all)")
    ap.add_argument("--mb", type=int, default=8, help="micro-batch (sequences per forward)")
    ap.add_argument("--damp", type=float, default=0.01, help="GPTQ Hessian damping")
    ap.add_argument("--min-tokens", type=int, default=64,
                    help="minimum routed tokens for GPTQ; below -> RTN fallback")
    ap.add_argument("--mse-scale", choices=["on", "off"], default="on",
                    help="per-row NVFP4 block-scale search (default on)")
    ap.add_argument("--resume", action="store_true", help="skip finished layers, reload boundary hidden states")
    ap.add_argument("--assemble", action="store_true", help="write the final pack from per-layer outputs + the RTN pack")
    ap.add_argument("--selftest", action="store_true", help="run selftests (a), (b), (c); no full model")
    args = ap.parse_args()
    if args.selftest:
        selftest(args)
        return
    if args.assemble:
        assemble(args)
        return
    run(args)


if __name__ == "__main__":
    main()
