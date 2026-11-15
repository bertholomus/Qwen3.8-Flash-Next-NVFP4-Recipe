#!/usr/bin/env python3
# q38fn_kld.py — layerwise KLD harness: NVFP4 pack vs BF16 source.
# Fails loudly (RuntimeError naming the step) on any structural mismatch.
import argparse, hashlib, importlib.util, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda"
BIG = 1_500_000_000  # bytes; tensors larger than this (bf16) are treated as PLE tables
# Reference-arm switch. With KLD_PLE_REF=bf16 the reference arm reads the n-gram rows
# in true BF16 (no NVFP4 rounding), so the KL includes the n-gram table quant error. Default keeps old behaviour.
_PLE_ARM = {"mode": "q"}
# where the converter module (q38fn_convert.py) is looked up
_TOOLS_DIR = os.environ.get("CONVERTER_DIR", os.path.dirname(os.path.abspath(__file__)))

def fail(step, msg):
    raise RuntimeError(f"[FAIL step={step}] {msg}")

# ---------------------------------------------------------------- safetensors
def st_index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(p):
        fail("index", f"missing index {p}")
    with open(p) as f:
        return json.load(f)["weight_map"]

def st_open(d, shard):
    from safetensors import safe_open
    return safe_open(os.path.join(d, shard), framework="pt", device="cpu")

def st_load(d, index, name):
    if name not in index:
        fail("load_tensor", f"tensor {name!r} not in index of {d}")
    with st_open(d, index[name]) as f:
        return f.get_tensor(name)

_ST_HDR = {}
def st_memmap(d, name):
    """numpy memmap view of one tensor's data region (for huge PLE tables)."""
    idx = st_index(d)
    path = os.path.join(d, idx[name])
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(8), "<u8")[0])
        hdr = json.loads(f.read(n))
    if name not in hdr:
        fail("memmap", f"tensor {name!r} not in header of {path}")
    meta = hdr[name]
    dt = {"BF16": np.dtype("uint16"), "F32": np.dtype("f4"), "F16": np.dtype("f2"),
          "U8": np.dtype("u1"), "I32": np.dtype("i4"), "I64": np.dtype("i8"),
          "F8_E4M3": np.dtype("u1")}.get(meta["dtype"])
    if dt is None:
        fail("memmap", f"unsupported dtype {meta['dtype']} for {name}")
    a, b = meta["data_offsets"]
    mm = np.memmap(path, dtype=dt, mode="r", offset=8 + n + a, shape=tuple(meta["shape"]))
    return mm

def st_memmap_shape(d, name):
    """(shape, dtype-tag) from the safetensors header without opening the data."""
    idx = st_index(d)
    path = os.path.join(d, idx[name])
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(8), "<u8")[0])
        hdr = json.loads(f.read(n))
    if name not in hdr:
        fail("memmap", f"tensor {name!r} not in header of {path}")
    return tuple(hdr[name]["shape"]), hdr[name]["dtype"]

# ---------------------------------------------------------------- NVFP4 math
_E2M1 = torch.tensor([0.0, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

def fp8_e4m3_to_f32(x: torch.Tensor) -> torch.Tensor:
    # Non-FP8 inputs (e.g. the F32 scalar weight_scale_2) pass through as plain floats.
    if x.dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        return x.to(torch.float32)
    x = x.view(torch.uint8).to(torch.int32)
    s = torch.where((x >> 7) & 1 == 1, -1.0, 1.0)
    e = (x >> 3) & 0xF
    m = (x & 7).to(torch.float32)
    v = torch.where(e > 0, (1.0 + m / 8.0) * torch.pow(2.0, (e - 7).to(torch.float32)),
                    m / 8.0 * 2.0 ** -6)
    v = torch.where((e == 15) & (m == 7), torch.tensor(float("nan")), v)
    return s * v

def f32_to_fp8_e4m3(v: torch.Tensor) -> torch.Tensor:
    v = v.to(torch.float32)
    sign = (torch.signbit(v) & ~torch.isnan(v)).to(torch.int32) << 7
    a = v.abs()
    e = torch.floor(torch.log2(a.clamp_min(1e-30))).to(torch.int32).clamp(-6, 8)
    e = torch.where(a > 0, e, torch.full_like(e, -7))
    m = torch.round((a / torch.pow(2.0, e.to(torch.float32)) - 1.0) * 8.0).to(torch.int32)
    ovf = m > 15
    e = torch.where(ovf, e + 1, e); m = torch.where(ovf, torch.zeros_like(m), m)
    e = e.clamp(-6, 8)
    sub = e <= -6
    e_enc = torch.where(sub, torch.zeros_like(e), (e + 7).clamp(0, 15))
    m_enc = torch.where(sub, torch.round(a / 2.0 ** -6 * 8.0).to(torch.int32).clamp(0, 7), m)
    code = sign | (e_enc << 3) | m_enc.clamp(0, 7)
    code = torch.where(torch.isnan(v), torch.full_like(code, 0x7F), code)
    return code.clamp(0, 255).to(torch.uint8)

def e2m1_rne(v: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even quantization to e2m1 grid values."""
    pos = _E2M1[:8].to(v.device)
    a = v.abs()
    idx = torch.bucketize(a, torch.tensor([0.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=v.device))
    q = pos[idx.clamp(0, 7)]
    return torch.where(torch.signbit(v), -q, q)

def dequant_nvfp4(w_packed: torch.Tensor, scales: torch.Tensor, scale2: torch.Tensor,
                  out_shape) -> torch.Tensor:
    """weight = e2m1(nibble) * fp8_e4m3 block scale (16/block) * scale2, -> bf16."""
    step = "dequant_nvfp4"
    if w_packed.dtype != torch.uint8:
        fail(step, f"expected uint8 packed weights, got {w_packed.dtype}")
    flat = w_packed.reshape(-1)
    lo = (flat & 0xF).to(torch.int32); hi = (flat >> 4).to(torch.int32)
    nib = torch.stack([lo, hi], dim=-1).reshape(-1)
    if nib.numel() != math.prod(out_shape):
        fail(step, f"packed nibbles {nib.numel()} != target elems {math.prod(out_shape)} "
                   f"(layout mismatch vs converter — check q38fn_convert.py)")
    vals = _E2M1.to(w_packed.device)[nib].reshape(out_shape)
    sc = fp8_e4m3_to_f32(scales.to(w_packed.device)).to(torch.float32) * \
        fp8_e4m3_to_f32(scale2.to(w_packed.device)).to(torch.float32).reshape(())
    k = out_shape[-1]
    if k % 16 != 0:
        fail(step, f"last dim {k} not divisible by 16")
    sc = sc.reshape(out_shape[:-1] + (k // 16,))
    sc = sc.repeat_interleave(16, dim=-1)
    return (vals * sc).to(torch.bfloat16)

def dequant_codes_nvfp4(w_packed: torch.Tensor, scales: torch.Tensor, scale2: torch.Tensor,
                        out_shape):
    """Like dequant_nvfp4 but returns (codes, scaled bf16 values) for round-trip checks."""
    flat = w_packed.reshape(-1)
    lo = (flat & 0xF).to(torch.int64); hi = (flat >> 4).to(torch.int64)
    nib = torch.stack([lo, hi], dim=-1).reshape(-1)
    codes = nib.reshape(out_shape)
    vals = _E2M1.to(w_packed.device)[nib].reshape(out_shape)
    sc = fp8_e4m3_to_f32(scales.to(w_packed.device)).to(torch.float32) * \
        fp8_e4m3_to_f32(scale2.to(w_packed.device)).to(torch.float32).reshape(())
    k = out_shape[-1]
    sc = sc.reshape(out_shape[:-1] + (k // 16,)).repeat_interleave(16, dim=-1)
    return codes, (vals * sc).to(torch.bfloat16).to(torch.float32)

def dequant_mxfp8(w_codes: torch.Tensor, scales: torch.Tensor, out_shape):
    """MXFP8 dense variant (group 32): weight = e4m3 code * 2**(u8 scale byte - 127) -> bf16.
    Scale byte is UE8M0 (TensorFold zig/src/cuda/mxfp8.md); byte 255 = NaN per spec -> rejected."""
    step = "dequant-mxfp8"
    if w_codes.dtype != torch.float8_e4m3fn:
        fail(step, f"expected float8_e4m3fn weights, got {w_codes.dtype}")
    if scales.dtype != torch.uint8:
        fail(step, f"expected uint8 MXFP8 scales, got {scales.dtype}")
    n, k = out_shape[-2], out_shape[-1]
    if k % 32 != 0:
        fail(step, f"last dim {k} not divisible by 32")
    if tuple(w_codes.shape) != tuple(out_shape):
        fail(step, f"MXFP8 weight shape {tuple(w_codes.shape)} != source shape {tuple(out_shape)}")
    if tuple(scales.shape) != (n, k // 32):
        fail(step, f"MXFP8 scale shape {tuple(scales.shape)} != {(n, k // 32)}")
    if bool((scales == 255).any()):
        fail(step, "MXFP8 scale byte 255 (NaN) present")
    sc = torch.pow(2.0, (scales.to(torch.int32) - 127).to(torch.float32))
    vals = w_codes.to(torch.float32) * sc.repeat_interleave(32, dim=-1)
    return vals.to(torch.bfloat16).reshape(out_shape)

def quant_nvfp4_ref(w: torch.Tensor, s2_override=None):
    """Independent round-trip matching the converter: per-tensor s2 = global amax/(6*448)
    (or the pack's stored s2 when given), block scale = e4m3(amax_b/6/s2),
    codes = nearest E2M1 of w/dec (ties -> even code)."""
    k = w.shape[-1]
    blocks = w.reshape(-1, 16)
    amax = blocks.abs().amax(dim=-1).to(torch.float32)
    if s2_override is not None:
        s2 = s2_override.to(torch.float32).reshape(()).cpu() if torch.is_tensor(s2_override) \
            else torch.tensor(float(s2_override))
    else:
        s2 = (amax.max() / (6.0 * 448.0)).to(torch.float32)
    stored = (amax / 6.0 / s2.clamp(min=1e-30)).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
    sc = stored.to(torch.float32).reshape(-1, 1)
    dec = sc * s2
    pos = dec > 0
    ratio = blocks.to(torch.float32) / torch.where(pos, dec, torch.ones_like(dec))
    mid = torch.tensor([0.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=w.device)
    vals = torch.tensor([0.0, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    m = ratio.abs().clamp(0.0, 6.0)
    idx = torch.searchsorted(mid, m.contiguous(), right=True)
    lo = (idx - 1).clamp(min=0); hi = idx.clamp(max=7)
    dlo = m - vals[lo]; dhi = vals[hi] - m
    even = lo + (lo & 1)
    code = torch.where(dlo < dhi, lo, torch.where(dhi < dlo, hi, even)).to(torch.uint8)
    code = torch.where((amax > 0).unsqueeze(-1), code, torch.zeros_like(code))
    code = code | ((ratio < 0).to(torch.uint8) * 8)
    return code.reshape(w.shape), sc.reshape(w.shape[:-1] + (k // 16,))

# ---------------------------------------------------------------- converters
def load_converter(path):
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("q38fn_convert", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        fail("converter-import", f"could not import {path}: {e}")
    return mod

# ---------------------------------------------------------------- model bits
def build_config(src):
    step = "config"
    try:
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig as C
    except Exception:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(src)
            tc = getattr(cfg, "text_config", None)
            return tc if tc is not None else cfg
        except Exception as e:
            fail(step, f"cannot load config: {e}")
    cfg = C.from_pretrained(src)
    # Qwen4ExpTextModel expects the TEXT config (vocab_size lives in text_config).
    tc = getattr(cfg, "text_config", None)
    return tc if tc is not None else cfg

def build_model_parts(src, cfg):
    step = "import-model"
    try:
        import transformers.models.qwen4_exp.modeling_qwen4_exp as M
    except Exception as e:
        fail(step, f"cannot import modeling_qwen4_exp: {e}")
    with torch.device("meta"):
        model = M.Qwen4ExpTextModel(cfg)
    parts = {"embed": model.embed_tokens, "layers": list(model.layers),
             "mixer": model.hyper_connection_mixer,
             "norm": getattr(model.hyper_connection_mixer, "hc_norm", None),
             "lm_head": None, "M": M}
    if parts["mixer"] is None:
        fail("final-path", "Qwen4ExpTextModel has no hyper_connection_mixer")
    if parts["norm"] is None:
        fail("final-path", "cannot locate final norm on hyper_connection_mixer — read the source")
    try:
        lm = M.Qwen4ExpTextForCausalLM if hasattr(M, "Qwen4ExpTextForCausalLM") else None
    except Exception:
        lm = None
    return parts, M

def clone_meta(mod):
    import copy
    return copy.deepcopy(mod)

def materialize(mod):
    mod.to_empty(device=DEV)
    mod.to(torch.bfloat16)   # checkpoint weights are BF16; default init is fp32
    return mod

# ------------------------------------------------------- PLE lazy gather patch
def patch_ple(layer, pack_dir, pack_index, prefix, src_dir, src_index):
    """Shrink the meta-built ngram_embedding to a 1-row dummy so to_empty()
    does not try to allocate the ~95 GiB meta table on GPU. Must run on the
    META module BEFORE materialize(clone_meta(...))."""
    step = "ple-shrink"
    full_prefix = prefix.replace("model.", "model.language_model.", 1)
    targets = [(n, m) for n, m in layer.named_modules()
               if hasattr(m, "weight") and "ngram_embedding" in n and isinstance(m, nn.Embedding)]
    if not targets:
        return
    for name, emb in targets:
        emb.weight = nn.Parameter(torch.zeros(1, emb.weight.shape[1],
                                              dtype=emb.weight.dtype), requires_grad=False)
        emb.weight.meta_like_marker = True

def patch_ple_hooks(layer, pack_dir, pack_index, prefix, src_dir, src_index):
    """Attach the lazy dequantizing PLE forward to a MATERIALIZED layer.

    The pack stores the table as shard_<i>.weight (U8 [rows,80] packed E2M1),
    shard_<i>.weight_scale (e4m3 [rows,10]) and ONE shared F32 weight_scale_2.
    The BF16 source stores the same rows as shard_<i>.weight [rows,160].
    We keep the source rows on disk (memmap, uint16) and dequantize only the
    gathered rows at forward time — mathematically identical to dequantizing
    the pack (verified bit-exact in selftest-b), without materialising ~28 GB.
    """
    step = "ple-patch"
    full_prefix = prefix.replace("model.", "model.language_model.", 1)
    owners = [(n, m) for n, m in layer.named_modules()
              if hasattr(m, "ngram_embedding") and isinstance(getattr(m, "ngram_embedding", None), nn.Embedding)]
    if not owners:
        return
    # the per-table shared scale lives at the layer that owns the PLE table
    # (config ple_layer_ids=[2] -> source layer index 1 in the index names)
    s2_name = f"{full_prefix}.ple.ple_embedding.ngram_embedding.weight_scale_2"
    if s2_name not in pack_index:
        fail(step, f"ngram shared scale {s2_name} missing from pack index")
    s2 = st_load(pack_dir, pack_index, s2_name).to(torch.float32).reshape(())
    for name, owner in owners:
        base = f"{full_prefix}.{name}.ngram_embedding"
        src_prefix = f"{base}.shard_"
        src_shards = sorted((int(n.split("shard_")[1].split(".")[0]), n)
                            for n in src_index if n.startswith(src_prefix) and n.endswith(".weight"))
        if not src_shards:
            fail(step, f"no source shards found for {base}")
        widths = {st_memmap_shape(src_dir, n)[0][1] for _, n in src_shards}
        if widths != {160}:
            fail(step, f"unexpected source shard widths {widths} for {base}")
        host = StShardedBF16(src_dir, src_index, [n for _, n in src_shards])

        def lazy_forward(ids, _host=host, _s2=float(s2.item()), _nm=base):
            ids_c = ids if ids.device == torch.device("cpu") else ids.cpu().to(torch.long)
            flat = ids_c.reshape(-1)
            rows = _host.gather(flat).to(DEV).to(torch.float32)  # [n,160] bf16->f32
            if _PLE_ARM["mode"] == "ref" and os.environ.get("KLD_PLE_REF") == "bf16":
                return rows.to(torch.bfloat16).reshape(*ids.shape, 160).flatten(-2)
            if _PLE_ARM["mode"] == "q" and os.environ.get("KLD_PLE_Q") == "bf16":
                # Control check: q arm also reads true BF16 rows -> KL must be exactly 0 (toggle check).
                return rows.to(torch.bfloat16).reshape(*ids.shape, 160).flatten(-2)
            if _PLE_ARM["mode"] == "q" and os.environ.get("KLD_PLE_Q") == "fp8b32":
                # Diagnostic: FP8 E4M3 rows with one scale per 32 values (5 scales per 160-wide row).
                # Tells whether the per-row FP8 loss (KL 0.037) is scale granularity or table sensitivity.
                blk = rows.reshape(-1, 32)
                sc = (blk.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
                dec = (blk / sc).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32) * sc
                return dec.reshape(-1, 160).to(torch.bfloat16).reshape(*ids.shape, 160).flatten(-2)
            if _PLE_ARM["mode"] == "q" and os.environ.get("KLD_PLE_Q") == "fp8":
                # Diagnostic: q arm reads the n-gram rows as FP8 E4M3 with one F32 scale per row
                # (amax/448). Estimates the cost of FP8 rows (TensorFold reads fp8 rows) vs NVFP4 rows. Default unchanged.
                sc = (rows.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
                dec = (rows / sc).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32) * sc
                return dec.to(torch.bfloat16).reshape(*ids.shape, 160).flatten(-2)
            # dequantize like the pack: per-16 amax -> e4m3 scale * s2, then E2M1 values
            blk = rows.reshape(-1, 16)
            amax = blk.abs().amax(-1)
            stored = (amax / 6.0 / max(_s2, 1e-30)).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
            dec = (stored.to(torch.float32) * _s2).clamp_min(1e-30).unsqueeze(-1)
            ratio = blk / dec
            vals = e2m1_rne(ratio) * dec
            out = vals.to(torch.bfloat16).reshape(*ids.shape, 160)
            return out.flatten(-2)  # same contract as ngram_embedding forward output

        def owner_forward(input_ids, past_key_values=None, _owner=owner, _lf=lazy_forward):
            # replicate Qwen4ExpTextNGramEmbedding.forward hash math; replace only
            # the final self.ngram_embedding(...) call with the lazy gather.
            cfg = _owner
            input_ids = input_ids.long()
            prev = input_ids.new_full((input_ids.shape[0], cfg.context_len), cfg.eos_token_id)
            token_history = torch.cat([prev, input_ids], dim=-1)
            shifted = [cfg._shift_right_ignore_eos(token_history, sh)
                       for sh in range(cfg.ngram_size)]
            blocks = []
            for ngram in range(2, cfg.ngram_size + 1):
                start_idx = (ngram - 2) * cfg.heads_per_ngram
                end_idx = start_idx + cfg.heads_per_ngram
                mixed_ids = shifted[0] * cfg.layer_multipliers[0]
                for position in range(1, ngram):
                    mixed_ids = torch.bitwise_xor(
                        mixed_ids, shifted[position] * cfg.layer_multipliers[position])
                hvs = cfg.ngram_heads_vocab_sizes[start_idx:end_idx]
                hoff = cfg.ngram_heads_offsets[start_idx:end_idx]
                nids = torch.remainder(mixed_ids.unsqueeze(-1), hvs.view(1, 1, -1))
                blocks.append(nids + hoff.view(1, 1, -1))
            ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]
            # _lf returns [.., heads, head_dim]; original flattens to [.., heads*head_dim]
            return _lf(ngram_ids)
        owner.forward = owner_forward
        emb = owner.ngram_embedding
        emb.weight = nn.Parameter(torch.empty(1, emb.weight.shape[1],
                                              dtype=torch.bfloat16, device=DEV),
                                  requires_grad=False)

class StShardedBF16:
    """Lazy row-gather over the BF16 source shards (uint16 memmaps)."""
    def __init__(self, src_dir, src_index, shard_names):
        self.maps = [torch.from_numpy(np.asarray(st_memmap(src_dir, n))) for n in shard_names]
        self.width = self.maps[0].shape[1]
        self.offsets = []
        off = 0
        for mp in self.maps:
            self.offsets.append(off)
            off += mp.shape[0]
        self.total = off

    def gather(self, ids):
        # int64 ids may exceed int32 for 320M-row tables; use a Python-side mask loop
        # but keep each shard lookup as a plain uint16 numpy slice -> torch (no bf16 copy).
        n = ids.numel()
        out_np = np.empty((n, self.width), dtype=np.uint16)
        done = np.zeros(n, dtype=bool)
        ids_np = ids.numpy() if ids.device.type == "cpu" else ids.cpu().numpy()
        for mp, off in zip(self.maps, self.offsets):
            lo, hi = off, off + mp.shape[0]
            sel = (ids_np >= lo) & (ids_np < hi) & ~done
            if sel.any():
                local = ids_np[sel].astype(np.int64) - lo
                out_np[sel] = np.asarray(mp[local])
                done |= sel
            if bool(done.all()):
                break
        if not bool(done.all()):
            bad = ids_np[~done][:8].tolist()
            raise RuntimeError(f"[FAIL step=ple-patch] row ids out of table range: {bad}")
        # uint16 bit pattern -> bf16 values
        return torch.from_numpy(out_np).view(torch.bfloat16)

# ------------------------------------------------------- weight loading
def load_layer_bf16(layer, src_dir, src_index, prefix, step="load-bf16"):
    # Source index uses the multimodal layout "model.language_model.<...>";
    # the text-only module tree uses "model.<...>".
    full_prefix = prefix.replace("model.", "model.language_model.", 1)
    sd = {}
    for name in src_index:
        if name.startswith(full_prefix + "."):
            rel = name[len(full_prefix) + 1:]
            if ".ngram_embedding.shard_" in rel:
                continue  # PLE table is served by the lazy gather, not state_dict
            sd[rel] = st_load(src_dir, src_index, name)
    res = layer.load_state_dict(sd, strict=False)
    # the ONLY allowed missing key is the PLE table (replaced by the lazy gather)
    missing = list(res.missing_keys or [])
    unexpected = list(res.unexpected_keys or [])
    if missing:
        allowed = {"ple.ple_embedding.ngram_embedding.weight"}
        bad = [str(mm) for mm in missing if str(mm) not in allowed]
        if bad:
            fail(step, f"strict load missing keys: {bad}")
    if unexpected:
        fail(step, f"strict load unexpected keys: {sorted(str(u) for u in unexpected)[:8]}")
    return sd

def load_layer_pack(layer, pack_dir, pack_index, src_dir, src_index, prefix, conv):
    """Load pack weights into module layout: dequant NVFP4, verify BF16 identity."""
    step = "load-pack"
    full_prefix = prefix.replace("model.", "model.language_model.", 1)
    # Pack reuses the source multimodal layout "model.language_model.<...>".
    # Experts: the converter splits the fused [512,*] source tensors into per-expert
    # pack tensors (mlp.experts.<e>.<proj>.weight); re-fuse them for the module.
    GU = full_prefix + ".mlp.experts.gate_up_proj"
    DN = full_prefix + ".mlp.experts.down_proj"
    if GU not in pack_index:
        # pack keys are per-expert: probe experts.0.gate_proj.weight instead
        GU_probe = f"{full_prefix}.mlp.experts.0.gate_proj.weight"
        if GU_probe not in pack_index:
            fail(step, f"no expert tensors found in pack for {full_prefix}")
    fused = {}
    sd_fused = {}
    num_experts = None
    for e in range(1024):
        if f"{full_prefix}.mlp.experts.{e}.gate_proj.weight" not in pack_index:
            num_experts = e
            break
    if not num_experts:
        fail(step, f"could not count experts for {full_prefix}")
    parts = {p: [] for p in ("gate_proj", "up_proj", "down_proj")}
    for p in ("gate_proj", "up_proj", "down_proj"):
        for e in range(num_experts):
            base = f"{full_prefix}.mlp.experts.{e}.{p}"
            t = st_load(pack_dir, pack_index, base + ".weight")
            if t.dtype != torch.uint8:
                fail(step, f"expert tensor {base}.weight dtype {t.dtype} != uint8")
            scales = st_load(pack_dir, pack_index, base + ".weight_scale")
            s2name = base + ".weight_scale_2"
            if s2name not in pack_index:
                fail(step, f"packed tensor {base}.weight lacks per-tensor scale {s2name}")
            s2 = st_load(pack_dir, pack_index, s2name)
            if p == "down_proj":
                out_shape = (2560, 640)
            else:
                out_shape = (640, 2560)
            parts[p].append(dequant_nvfp4(t, scales, s2, out_shape))
            del t, scales, s2
    gu = torch.cat([torch.stack(parts["gate_proj"]), torch.stack(parts["up_proj"])], dim=1)
    dn = torch.stack(parts["down_proj"])
    del parts
    sd_fused = {"mlp.experts.gate_up_proj": gu, "mlp.experts.down_proj": dn}
    torch.cuda.empty_cache()
    sd = {}
    for rel in list(dict.fromkeys(
            [n[len(full_prefix) + 1:] for n in pack_index if n.startswith(full_prefix + ".")])):
        if rel.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
            continue
        if rel.startswith("mlp.experts."):
            continue
        if ".ngram_embedding.shard_" in rel:
            continue  # PLE table handled by patch_ple (lazy dequant gather)
        if rel.startswith("ple.ple_embedding.layer_multipliers") or \
                rel.startswith("ple.ple_embedding.ngram_heads_"):
            continue  # PLE buffers: the lazy owner_forward uses the module's own (bit-exact)
        full = f"{full_prefix}.{rel}"
        t = st_load(pack_dir, pack_index, full)
        if t.dtype == torch.bfloat16 or t.dtype in (torch.float16, torch.float32):
            # BF16-kept tensor: must be bit-identical to source
            if full not in src_index:
                fail(step, f"BF16 pack tensor {full} not found in source for identity check")
            s = st_load(src_dir, src_index, full)
            if t.dtype != s.dtype:
                t = t.to(s.dtype)
            if not torch.equal(t.cpu(), s.cpu()):
                fail(step, f"BF16 tensor {full} is NOT bit-identical to source")
            sd[rel] = s
            del s
        elif t.dtype == torch.uint8:
            sname = full + "_scale"
            if sname not in pack_index:
                fail(step, f"packed tensor {full} lacks scale tensor {sname} in pack index")
            scales = st_load(pack_dir, pack_index, sname)
            s2name = full + "_scale_2"
            if s2name not in pack_index:
                fail(step, f"packed tensor {full} lacks per-tensor scale {s2name}")
            s2 = st_load(pack_dir, pack_index, s2name)
            # target shape: from source tensor of same role
            if full not in src_index:
                fail(step, f"no source tensor {full} to infer dequant shape (expert fusion mapping?)")
            shape = tuple(st_load(src_dir, src_index, full).shape)
            sd[rel] = dequant_nvfp4(t, scales, s2, shape)
            del t, scales, s2
        elif t.dtype == torch.float8_e4m3fn:
            # MXFP8 dense variant: <proj>.weight F8_E4M3 [n,k] +
            # <proj>.weight_scale U8 [n,k/32] UE8M0 byte, scale = 2**(byte-127).
            sname = full + "_scale"
            if sname not in pack_index:
                fail(step, f"packed tensor {full} lacks scale tensor {sname} in pack index")
            scales = st_load(pack_dir, pack_index, sname)
            if full + "_scale_2" in pack_index:
                fail(step, f"MXFP8 tensor {full} unexpectedly has {full}_scale_2 in pack index")
            if full not in src_index:
                fail(step, f"no source tensor {full} to infer dequant shape")
            shape = tuple(st_load(src_dir, src_index, full).shape)
            sd[rel] = dequant_mxfp8(t, scales, shape)
            del t, scales
        else:
            fail(step, f"unsupported pack dtype {t.dtype} for {full}")
    sd.update(sd_fused)
    # expert split/fusion: if module param is [E,...] fused but pack stored per-expert, or vice versa
    cur = {k: v for k, v in layer.state_dict().items()}
    fixed = {}
    for k, v in sd.items():
        if k in cur and cur[k].shape == v.shape:
            fixed[k] = v
        else:
            fail(step, f"shape mismatch for {prefix}.{k}: module {tuple(cur[k].shape) if k in cur else 'MISSING'} vs pack {tuple(v.shape)}")
    res = layer.load_state_dict(fixed, strict=False)
    # the ONLY allowed missing keys are the PLE table + buffers (lazy owner_forward path)
    missing = list(res.missing_keys or [])
    unexpected = list(res.unexpected_keys or [])
    allowed = {"ple.ple_embedding.layer_multipliers", "ple.ple_embedding.ngram_heads_vocab_sizes",
               "ple.ple_embedding.ngram_heads_offsets", "ple.ple_embedding.ngram_embedding.weight"}
    if missing:
        bad = [str(mm) for mm in missing if str(mm) not in allowed]
        if bad:
            fail(step, f"pack load missing keys: {bad}")
    if unexpected:
        fail(step, f"pack load unexpected keys: {sorted(str(u) for u in unexpected)[:8]}")
    del sd, fixed, sd_fused
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- eval data
def build_tokens(tok, docs, seq_len, nseq):
    ids = []
    for d in docs:
        ids.extend(tok(d["text"], add_special_tokens=False)["input_ids"])
        while len(ids) >= seq_len:
            break
        if len(ids) >= seq_len * nseq:
            break
    if len(ids) < seq_len * nseq:
        fail("eval-data", f"eval text too short: need {seq_len*nseq} tokens, got {len(ids)}")
    ids = ids[: seq_len * nseq]
    return torch.tensor(ids, dtype=torch.long).reshape(nseq, seq_len)

# ---------------------------------------------------------------- main run
@torch.no_grad()
def run(args):
    t0 = time.time()
    tok_dir = args.src
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    docs = [json.loads(l) for l in open(args.eval)]
    ids = build_tokens(tok, docs, args.seq_len, args.nseq)
    sha = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    ids = ids.to(DEV)
    cfg = build_config(args.src)
    parts, M = build_model_parts(args.src, cfg)
    src_index, pack_index = st_index(args.src), st_index(args.pack)
    conv = load_converter(os.path.join(_TOOLS_DIR, "q38fn_convert.py"))

    n_layers = args.layers if args.layers else len(parts["layers"])
    B, T = args.nseq, args.seq_len
    width = cfg.hc_count * cfg.hidden_size
    h_ref = torch.zeros(B, T, width, dtype=torch.bfloat16, device=DEV)
    h_q = torch.zeros_like(h_ref)
    rel_errs = []
    mb = max(1, int(os.environ.get("KLD_MB", "8")))

    # embedding (BF16 in both, must be identical); the residual stream is
    # hc_count x hidden (hyper-connections): embed output is repeated like
    # Qwen4ExpTextModel.forward does (hidden.repeat(1, 1, hc_count)).
    emb_mod = materialize(clone_meta(parts["embed"]))
    load_layer_bf16(emb_mod, args.src, src_index, "model.embed_tokens")
    for s in range(0, B, mb):
        h_ref[s:s+mb] = emb_mod(ids[s:s+mb]).to(torch.bfloat16).repeat(1, 1, cfg.hc_count)
    load_layer_bf16(emb_mod, args.src, src_index, "model.embed_tokens")  # ensure same
    h_q.copy_(h_ref)  # embedding identical in both paths
    del emb_mod; torch.cuda.empty_cache()

    # rotary embeddings: computed on the fly (no weights in the checkpoint);
    # build the class directly from the text config.
    rotary = parts["M"].Qwen4ExpTextRotaryEmbedding(cfg)
    rotary = rotary.to(device=DEV)

    pos = torch.arange(T, device=DEV).unsqueeze(0).expand(B, -1)
    position_embeddings = rotary(torch.zeros(B, 1, cfg.hidden_size, dtype=torch.bfloat16, device=DEV),
                                 torch.arange(T, device=DEV).unsqueeze(0))
    # Fix: rotary returns batch 1; the indexed-attention indexer indexes cos/sin per batch row
    # (modeling_qwen4_exp.py:740 full_cos[batch_idx]); expand to B so every micro-batch slice has real rows.
    position_embeddings = tuple(t.expand(B, *t.shape[1:]).contiguous() for t in position_embeddings)
    M = parts["M"]
    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        meta_layer = parts["layers"][i]
        patch_ple(meta_layer, args.pack, pack_index, prefix, args.src, src_index)  # shrink meta embed
        layer = materialize(clone_meta(meta_layer))
        patch_ple_hooks(layer, args.pack, pack_index, prefix, args.src, src_index)
        load_layer_bf16(layer, args.src, src_index, prefix)
        r_s = q_s = 0.0
        cfg._attn_implementation = "sdpa"
        _PLE_ARM["mode"] = "ref"
        for s in range(0, B, mb):
            pos_mb = pos[s:s+mb]  # (bs, T); create_*_mask expands internally
            # build the proper 4-D causal masks like Qwen4ExpTextModel.forward does
            mask_kwargs = {"config": cfg, "inputs_embeds": h_ref[s:s+mb], "attention_mask": None,
                           "past_key_values": None, "position_ids": pos_mb,
                           "allow_is_causal_skip": False}
            cm = M.create_causal_mask(**mask_kwargs)
            out = layer(h_ref[s:s+mb], position_ids=pos_mb, position_embeddings=tuple(t[s:s+mb] for t in position_embeddings),
                        ple_input_ids=ids[s:s+mb], attention_mask=cm)
            h_ref[s:s+mb] = out[0] if isinstance(out, tuple) else out
        load_layer_pack(layer, args.pack, pack_index, args.src, src_index, prefix, conv)
        if os.environ.get("KLD_DIAG") == "pack_bf16_experts":
            # Diagnostic: q arm = pack loader path, but routed experts reloaded from BF16 source.
            # Expected KL == 0 exactly. Nonzero => the q-arm wiring (not quantization) moves the result.
            _fp = prefix.replace("model.", "model.language_model.", 1) + ".mlp.experts."
            _sd = {n[len(prefix.replace("model.", "model.language_model.", 1)) + 1:]: st_load(args.src, src_index, n)
                   for n in src_index if n.startswith(_fp)}
            _r = layer.load_state_dict(_sd, strict=False)
            if _r.unexpected_keys or len(_sd) == 0:
                fail("diag", f"expert reload failed: n={len(_sd)} unexpected={list(_r.unexpected_keys)[:4]}")
            del _sd
        _PLE_ARM["mode"] = "q"
        for s in range(0, B, mb):
            pos_mb = pos[s:s+mb]
            mask_kwargs = {"config": cfg, "inputs_embeds": h_q[s:s+mb], "attention_mask": None,
                           "past_key_values": None, "position_ids": pos_mb,
                           "allow_is_causal_skip": False}
            cm = M.create_causal_mask(**mask_kwargs)
            out = layer(h_q[s:s+mb], position_ids=pos_mb, position_embeddings=tuple(t[s:s+mb] for t in position_embeddings),
                        ple_input_ids=ids[s:s+mb], attention_mask=cm)
            h_q[s:s+mb] = out[0] if isinstance(out, tuple) else out
        r_s = h_ref.float().pow(2).sum().sqrt().item()
        q_s = h_q.float().pow(2).sum().sqrt().item()
        rel = (h_ref.float() - h_q.float()).pow(2).sum().sqrt().item() / max(r_s, 1e-12)
        rel_errs.append(rel)
        print(f"layer {i}/{len(parts['layers'])} ref_s={r_s:.4e} q_s={q_s:.4e} resid_rel_err={rel:.4e}", flush=True)
        del layer; torch.cuda.empty_cache()

    # final path: mixer already includes its hc_norm; the text model's forward does
    #   hidden = self.hyper_connection_mixer(hidden)  (GatedResidual mixes + norms internally)
    # then lm_head. hc_norm is applied inside mixer.forward on the hyper_input, so the
    # norm module here is NOT applied again — we only use the mixer output.
    norm = materialize(clone_meta(parts["norm"]))
    # Note: materialize() zeroes the weights via to_empty();
    # run 1 used an UNINITIALIZED mixer in both arms -> ppl_ref 7,647 (broken final
    # path), so all KL/top1 numbers were measured against a broken reference.
    # Load the checkpoint mixer weights (hc_norm + input_mix down/up) exactly as
    # Qwen4ExpTextModel.forward applies them (hc_norm runs inside mixer.forward).
    mixer = materialize(clone_meta(parts["mixer"]))
    load_layer_bf16(mixer, args.src, src_index, "model.hyper_connection_mixer", step="final-path")
    if hasattr(mixer, "use_combine"):
        mixer.use_combine = False
    del norm
    lm_names = [n for n in src_index if "lm_head" in n]
    if not lm_names:
        fail("final-path", "lm_head.weight not found in source index")
    lm_w = st_load(args.src, src_index, lm_names[0]).to(DEV)
    if "lm_head.weight" in pack_index:
        pw = st_load(args.pack, pack_index, "lm_head.weight")
        if not torch.equal(pw.cpu(), st_load(args.src, src_index, lm_names[0]).cpu()):
            fail("final-path", "lm_head in pack not bit-identical to source")

    kl_all, agree, nll_r, nll_q, ntok = [], 0, 0.0, 0.0, 0
    targets = ids[:, 1:]
    for s in range(0, B, mb):
        for c in range(0, T, 512):
            sl = slice(c, min(c + 512, T))
            hr = mixer(h_ref[s:s+mb, sl])
            hq = mixer(h_q[s:s+mb, sl])
            lr = torch.log_softmax((hr.float() @ lm_w.t().float()), dim=-1)
            lq = torch.log_softmax((hq.float() @ lm_w.t().float()), dim=-1)
            if sl.stop - 1 == sl.start: break
            tgt = targets[s:s+mb, sl.start:sl.stop-1]
            lr, lq = lr[:, :-1], lq[:, :-1]
            p = lr.exp()
            kl = (p * (lr - lq)).sum(-1)
            kl_all.append(kl.reshape(-1))
            agree += (lr.argmax(-1) == lq.argmax(-1)).sum().item()
            nll_r += -lr.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            nll_q += -lq.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            ntok += tgt.numel()
    kl = torch.cat(kl_all)
    n = n_layers
    res = {"kl_mean": kl.mean().item(), "kl_median": kl.median().item(),
           "kl_p99": kl.quantile(0.99).item(), "kl_max": kl.max().item(),
           "top1_agree": 100.0 * agree / ntok,
           "ppl_ref": math.exp(nll_r / ntok), "ppl_q": math.exp(nll_q / ntok),
           "tokens": ntok, "eval_sha256": sha, "seq_len": args.seq_len, "nseq": args.nseq,
           "per_layer_resid_rel_err": rel_errs, "seconds": time.time() - t0}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_layer_resid_rel_err"}, indent=1))

# ---------------------------------------------------------------- selftest
@torch.no_grad()
def selftest(args):
    print("selftest (a) bf16==bf16 KL=0 ...")
    cfg = build_config(args.src); parts, M = build_model_parts(args.src, cfg)
    src_index, pack_index = st_index(args.src), st_index(args.pack)
    conv = load_converter(os.path.join(_TOOLS_DIR, "q38fn_convert.py"))
    # (c) strict load layer 0 and 3
    for i in (0, 3):
        layer = materialize(clone_meta(parts["layers"][i]))
        load_layer_bf16(layer, args.src, src_index, f"model.layers.{i}")
        print(f"  strict load layer {i}: OK")
        del layer
    # (b) round-trip dequant check on one expert tensor + one ngram block
    # The converter splits the fused source experts into per-expert pack tensors:
    #   src mlp.experts.gate_up_proj [512, 2560, 2560] -> pack experts.<e>.gate_proj/up_proj
    #   src mlp.experts.down_proj     [512, 2560, 640] -> pack experts.<e>.down_proj
    checked = 0
    for name in sorted(src_index):
        if not (".mlp.experts.gate_up_proj" in name or ".mlp.experts.down_proj" in name):
            continue
        s = st_load(args.src, src_index, name).to(torch.float32)  # [E, O, I]
        E, O, I = s.shape
        e0 = 0
        if ".gate_up_proj" in name:
            p0, p1, o0, o1 = "gate_proj", "up_proj", O // 2, O - O // 2
            pack_names = [(f"{name[:-len('.gate_up_proj')]}.{e0}.{p0}.weight", s[e0, :o0]),
                          (f"{name[:-len('.gate_up_proj')]}.{e0}.{p1}.weight", s[e0, o0:])]
        else:
            pack_names = [(f"{name[:-len('.down_proj')]}.{e0}.down_proj.weight", s[e0])]
        for pname, sw in pack_names:
            if pname not in pack_index:
                fail("selftest-b", f"expected pack tensor {pname} missing from pack index")
            t = st_load(args.pack, pack_index, pname)
            if t.dtype != torch.uint8:
                fail("selftest-b", f"pack tensor {pname} dtype {t.dtype} != uint8")
            scales = st_load(args.pack, pack_index, pname + "_scale")
            s2 = st_load(args.pack, pack_index, pname + "_scale_2")
            q, sc = quant_nvfp4_ref(sw, s2)
            pk_codes, deq = dequant_codes_nvfp4(t, scales, s2, tuple(sw.shape))
            q64 = q.to(torch.int64)
            if not torch.equal(pk_codes, q64):
                nbad = (pk_codes != q64).sum().item()
                fail("selftest-b", f"expert round-trip: {nbad} code mismatches for {pname} "
                                   "(converter layout/rounding differs — read q38fn_convert.py)")
            # ref scaled values: grid value x ref block scale, rounded to bf16 (as dequant does)
            dec_ref = (sc.to(torch.float32) * fp8_e4m3_to_f32(s2.to(torch.float32))
                       .reshape(())).repeat_interleave(16, dim=-1)
            ref_scaled = (_E2M1.to(deq.device)[q64 & 7] *
                          torch.where(q64 >= 8, -1.0, 1.0) * dec_ref).to(torch.bfloat16).to(torch.float32)
            d = (deq - ref_scaled).abs().max().item()
            if d != 0.0:
                fail("selftest-b", f"expert round-trip max abs diff {d} != 0 for {pname} "
                                   "(converter layout/rounding differs — read q38fn_convert.py)")
            print(f"  expert round-trip {pname}: max abs diff 0")
            checked += 1
        break
    if checked == 0:
        fail("selftest-b", "no packed expert tensor found to round-trip check")
    # ngram: pack shards are NVFP4 (U8 [rows,80] + e4m3 scales + shared s2) of the BF16
    # source shards ([rows,160]) — compare a dequantized block to the source rows.
    ng_pack = sorted(n for n in pack_index if ".ngram_embedding.shard_0.weight" in n)
    ng_src = sorted(n for n in src_index if ".ngram_embedding.shard_0.weight" in n)
    if not ng_pack or not ng_src:
        fail("selftest-b", f"ngram shard missing: pack={bool(ng_pack)} src={bool(ng_src)}")
    pname, sname = ng_pack[0], ng_src[0]
    mm = st_memmap(args.pack, pname)            # U8 [rows, 80]
    sc_mm = st_memmap(args.pack, pname + "_scale")  # e4m3 [rows, 10]
    src_mm = st_memmap(args.src, sname)         # BF16 [rows, 160] as uint16
    rows = min(4096, mm.shape[0])
    pk = torch.from_numpy(np.asarray(mm[:rows]).copy())
    sc = torch.from_numpy(np.asarray(sc_mm[:rows]).copy())
    s2_name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale_2"
    if s2_name not in pack_index:
        fail("selftest-b", f"ngram shared per-tensor scale {s2_name} missing from pack index")
    s2_np = st_load(args.pack, pack_index, s2_name)
    src_rows = torch.from_numpy(np.asarray(src_mm[:rows]).copy()) \
        .view(torch.bfloat16).to(torch.float32)
    deq = dequant_nvfp4(pk, sc, s2_np, (rows, src_mm.shape[1])).to(torch.float32)
    q_ref, sc_ref = quant_nvfp4_ref(src_rows, s2_np)
    q64 = q_ref.to(torch.int64)
    dec_ref = (sc_ref.to(torch.float32) * fp8_e4m3_to_f32(s2_np.to(torch.float32))
               .reshape(())).repeat_interleave(16, dim=-1)
    ref_scaled = (_E2M1.to(deq.device)[q64 & 7] *
                  torch.where(q64 >= 8, -1.0, 1.0) * dec_ref).to(torch.bfloat16).to(torch.float32)
    d = (deq - ref_scaled).abs().max().item()
    if d != 0.0:
        fail("selftest-b", f"ngram round-trip max abs diff {d} != 0 for {pname} "
                           "(see the n-gram table builder)")
    print(f"  ngram round-trip {pname}: max abs diff 0")
    print("SELFTEST PASS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./base-bf16")
    ap.add_argument("--pack", default="./pack-rtn")
    ap.add_argument("--eval", default="./kld_eval.jsonl")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--nseq", type=int, default=48)
    ap.add_argument("--out", default="./kld.json")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args); return
    run(args)

if __name__ == "__main__":
    main()
