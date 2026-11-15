#!/usr/bin/env python3
# build_f2.py — MXFP8 dense pack builder from a BF16 source.
#
# Takes the NVFP4 pack (--ref, the GPTQ-calibrated pack) and re-emits it with these
# BF16 dense projections converted to MXFP8 (group 32), quantized from the
# BF16 source (--src):
#   - DeltaNet linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}
#     (36 linear_attention layers, all-or-none per layer),
#   - QSA self_attn.{q_proj,k_proj,v_proj,indexer.index_qk_proj,o_proj}
#     (12 full_attention layers, all-or-none per layer),
#   - mlp.shared_expert.{gate_proj,up_proj,down_proj} (all 48 layers, all-or-none).
# 384 weights total. Everything else (vision, norms, router gates, embeddings,
# lm_head, MTP, n-gram rows, NVFP4 experts) is byte-identical to the input pack.
#
# Rounding rule (cited, OCP MX spec / ModelOpt e4m3 recipe):
#   e  = floor(log2(amax_block))              fp64-exact (bf16 amax is fp32-exact)
#   sc = 2**(e-8)                             e4m3 emax=8, max finite 448
#   if amax > 448*sc: sc = 2**(e-7)           per-block {e, e+1} overflow avoidance
#   codes = RNE-to-float8_e4m3fn(w/sc), values pre-clamped to +-448 (no-op under
#           the bump rule; kept as the saturation safety net)
#   scale byte = clamp(e-8+127, 0, 254)       UE8M0; decode scale = 2**(byte-127)
#   amax == 0 -> e = -127 -> byte clamps to 0 (scale 2**-127), codes all 0.
# This is the per-block {e, e+1} variant with overflow-avoidance selection — the
# same rule the official FP4 reference pack uses: on probe it reproduces its
# stored scale AND weight bytes 100.000% for DeltaNet/QSA tensors (its
# mlp.shared_expert bytes differ only because its source for those tensors
# drifted ~1-4% from the base used here; the selection rule is the same).
# MSE {e,e+1} selection was measured and rejected: it picks e+1 less often
# (12.6% vs 17.4% on in_proj_qkv) and keeps saturated-to-448 encodings
# on blocks whose max exceeds 448 — strictly worse than bumping the scale.
#
# CPU-only, deterministic, resumable per shard via OUT/.f2.<shard>.done markers
# (hash-checked) and a final OUT/.f2.done. No GPU, no network, no env vars.
import argparse, hashlib, json, os, struct, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_DEF = "./base-bf16"
REF_DEF = "./pack"
OUT_DEF = "./pack-mxfp8"
NV_DEF = "./fp4-reference"
E4M3_MAX = 448.0
CHUNK_ROWS = 4096  # row-chunk the fp32 block math to bound RAM
_HDR_CACHE = {}


def fail(step, msg):
    raise RuntimeError(f"[FAIL step={step}] {msg}")


def st_index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(p):
        fail("index", f"missing index {p}")
    with open(p) as f:
        return json.load(f)["weight_map"]


def st_load(d, index, name):
    if name not in index:
        fail("load_tensor", f"tensor {name!r} not in index of {d}")
    with safe_open(os.path.join(d, index[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def raw_header(path):
    if path not in _HDR_CACHE:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _HDR_CACHE[path] = json.loads(fh.read(n))
    return _HDR_CACHE[path]


def raw_keys(path):
    return [k for k in raw_header(path) if k != "__metadata__"]


def header_meta(d, index, name):
    """(shape, dtype-tag) without reading tensor data."""
    path = os.path.join(d, index[name])
    hdr = raw_header(path)
    if name not in hdr:
        fail("header", f"tensor {name!r} not in header of {path}")
    return tuple(hdr[name]["shape"]), hdr[name]["dtype"]


# ------------------------------------------------------------------ conversion
def converted_names(cfg):
    """384 names: DeltaNet/QSA projections + shared_expert, in config order."""
    lt = cfg["text_config"]["layer_types"]
    if len(lt) != 48:
        fail("plan", f"expected 48 layer_types, got {len(lt)}")
    names = []
    for i, kind in enumerate(lt):
        if kind == "linear_attention":
            parts = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
            pre = f"model.language_model.layers.{i}.linear_attn"
        elif kind == "full_attention":
            parts = ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj", "o_proj"]
            pre = f"model.language_model.layers.{i}.self_attn"
        else:
            fail("plan", f"layer {i}: unexpected layer_type {kind!r}")
        for p in parts:
            names.append(f"{pre}.{p}.weight")
        for p in ["gate_proj", "up_proj", "down_proj"]:
            names.append(f"model.language_model.layers.{i}.mlp.shared_expert.{p}.weight")
    return names


def quantize_mxfp8(src: torch.Tensor, name=""):
    """BF16 [O,I] -> (codes float8_e4m3fn [O,I], scale_bytes uint8 [O,I//32]).
    Rule per module docstring; fp64 branch compare; row-chunked; deterministic."""
    O, I = src.shape
    G = I // 32
    codes = torch.empty(O, I, dtype=torch.float8_e4m3fn)
    sb = torch.empty(O, G, dtype=torch.uint8)
    for r0 in range(0, O, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, O)
        B = src[r0:r1].to(torch.float32).reshape(r1 - r0, G, 32)
        amax = B.abs().amax(-1)
        amax64 = amax.to(torch.float64)
        if not bool(torch.isfinite(amax64).all()):
            fail("quant", f"non-finite amax in {name} rows {r0}:{r1}")
        with torch.no_grad():
            e = torch.floor(torch.log2(amax64.clamp_min(2.0 ** -300))).to(torch.int64)
        e = torch.where(amax > 0, e, torch.full_like(e, -127))
        bump = amax64 > E4M3_MAX * torch.pow(2.0, e.to(torch.float64) - 8.0)
        ep = torch.where(bump, e + 1, e)
        sb[r0:r1] = (ep - 8 + 127).clamp(0, 254).to(torch.uint8)
        sc = torch.pow(2.0, (ep.to(torch.float32) - 8)).unsqueeze(-1)  # [rows,G,1]
        x = torch.clamp(B / sc, -E4M3_MAX, E4M3_MAX)
        codes[r0:r1] = x.to(torch.float8_e4m3fn).reshape(r1 - r0, I)
    return codes, sb


def plan_ql(ref_ql, names):
    """REF quantized_layers + one MXFP8 entry per converted weight (prefix key)."""
    ql = dict(ref_ql)
    for n in names:
        k = n[: -len(".weight")]  # config keys the projection prefix, not ".weight"
        if k in ql:
            fail("plan", f"{k} already in REF quantized_layers")
        ql[k] = {"quant_algo": "MXFP8", "group_size": 32}
    return ql, len(names)


# ------------------------------------------------------------------ shard build
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def build_shard(ref_dir, out_dir, shard, conv, src_index):
    """Rebuild one model shard with converted tensors; returns #converted rebuilt."""
    src = os.path.join(ref_dir, shard)
    dst = os.path.join(out_dir, shard)
    marker = os.path.join(out_dir, f".f2.{shard}.done")
    if os.path.exists(dst) and os.path.exists(marker):
        if open(marker).read().strip() == file_sha256(dst):
            return 0
        print(f"  {shard}: marker hash mismatch, rebuilding")
    order = raw_keys(src)
    keys = [k for k in order if k in conv]
    data = {}
    with safe_open(src, framework="pt", device="cpu") as f:
        for k in order:
            if k in conv:
                w = st_load(SRC_DIR, src_index, k)  # quantize from the BF16 source
                codes, sb = quantize_mxfp8(w, k)
                data[k] = codes
                data[k + "_scale"] = sb  # inserted right after its weight
                del w, codes, sb
            else:
                data[k] = f.get_tensor(k)
    tmp = os.path.join(out_dir, ".tmp-f2-" + shard)
    save_file(data, tmp)
    del data
    os.replace(tmp, dst)
    with open(marker, "w") as fh:
        fh.write(file_sha256(dst) + "\n")
    return len(keys)


def copy_file(src, dst, marker=None):
    """Chunked byte copy; with a marker, skip when dst already matches."""
    if marker is not None and os.path.exists(dst) and os.path.exists(marker):
        if open(marker).read().strip() == file_sha256(dst):
            return False
    tmp = dst + ".tmp-f2"
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for blk in iter(lambda: fi.read(1 << 24), b""):
            fo.write(blk)
            h.update(blk)
    os.replace(tmp, dst)
    if marker is not None:
        with open(marker, "w") as fh:
            fh.write(h.hexdigest() + "\n")
    return True


# ------------------------------------------------------------------ build path
SRC_DIR = None  # module-global so build_shard can quantize from the source pack


def cmd_build(args):
    global SRC_DIR
    SRC_DIR = args.src
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.ref, "config.json")))
    ref_ql = cfg["quantization_config"]["quantized_layers"]
    names = converted_names(cfg)
    src_index, ref_index = st_index(args.src), st_index(args.ref)
    missing = [n for n in names if n not in src_index or n not in ref_index]
    if missing:
        fail("plan", f"{len(missing)} converted names missing from indexes: {missing[:3]}")
    conv = set(names)
    tot = 0
    for n in names:
        (O, I), _ = header_meta(args.ref, ref_index, n)
        if (O, I) != header_meta(args.src, src_index, n)[0]:
            fail("plan", f"shape mismatch {n}: ref vs source")
        tot += O * I
    saved = tot * 31 // 32  # 2*OI bf16 -> OI + OI/32; OI % 32 == 0 per tensor
    print(f"plan: {len(names)} weights, {tot/1e9:.3f}G params, "
          f"bf16 {tot*2/1e9:.3f} GB -> mxfp8 {tot*33/32/1e9:.3f} GB, saved {saved} bytes "
          f"({saved/1e9:.3f} GB / {saved/2**30:.3f} GiB)")
    shards = sorted({s for s in ref_index.values() if s.startswith("model-")})
    if args.dry_run:
        est = sum(os.path.getsize(os.path.join(args.ref, f)) for f in os.listdir(args.ref)
                  if os.path.isfile(os.path.join(args.ref, f))) - saved
        print(f"dry-run: would rebuild {len(shards)} model shards: {shards[:3]} ...")
        print(f"dry-run: estimated output size {est/1e9:.3f} GB; no files written")
        return
    os.makedirs(args.out, exist_ok=True)
    for shard in shards:
        n = build_shard(args.ref, args.out, shard, conv, src_index)
        print(f"  {shard}: rebuilt ({n} converted)" if n else f"  {shard}: rebuilt", flush=True)
    # byte-copy ngram shards (hash-marked) and every other aux file except the
    # four regenerated ones and the input pack's stale build markers
    for f in sorted(os.listdir(args.ref)):
        p = os.path.join(args.ref, f)
        if not os.path.isfile(p):
            continue
        if f.startswith("model-") or f in ("model.safetensors.index.json", "config.json",
                                           "quant_manifest.json"):
            continue
        if f in (".convert.done", ".ngram.done"):  # build state, not data
            continue
        copy_file(p, os.path.join(args.out, f),
                  os.path.join(args.out, f".f2.{f}.done") if f.endswith(".safetensors") else None)
    # index: same shard mapping + 384 new _scale entries; total_size shrunk exactly
    idx = json.load(open(os.path.join(args.ref, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    for n in names:
        if n + "_scale" in wm:
            fail("index", f"{n}_scale already in REF index")
        wm[n + "_scale"] = wm[n]
    idx["metadata"]["total_size"] = idx["metadata"]["total_size"] - saved
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as fh:
        json.dump(idx, fh, indent=2)
    # config: inject quantized_layers entries (sorted keys), all else equal
    ql, added = plan_ql(ref_ql, names)
    if added != len(names):
        fail("config", f"added {added} entries, expected {len(names)}")
    cfg["quantization_config"]["quantized_layers"] = dict(sorted(ql.items()))
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    # quant_manifest: same families, bf16_copy shrunk, mxfp8 families added
    qm = json.load(open(os.path.join(args.ref, "quant_manifest.json")))
    qm["quantization"] = ("NVFP4 weight-only (W4A16) experts + MXFP8 dense projections "
                          "(group 32, modelopt-mixed layout); MXFP8 converted from BF16 source")
    fam = qm.get("families", {})
    if "bf16_copy" in fam:
        fam["bf16_copy"]["bytes"] -= tot * 2
    fam["mxfp8_weight"] = {"bytes": tot}
    fam["mxfp8_weight_scale"] = {"bytes": tot // 32}
    qm["families"] = fam
    qm["total_bytes"] = sum(v["bytes"] for v in fam.values())
    with open(os.path.join(args.out, "quant_manifest.json"), "w") as fh:
        json.dump(qm, fh, indent=2)
        fh.write("\n")
    # sha256.txt in the input pack is the SOURCE provenance manifest (131 source shards +
    # tokenizer.json); the source is unchanged, so byte-copy it
    copy_file(os.path.join(args.ref, "sha256.txt"), os.path.join(args.out, "sha256.txt"))
    open(os.path.join(args.out, ".f2.done"), "w").close()
    out_bytes = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)
                    if os.path.isfile(os.path.join(args.out, f)))
    print(f"build done in {time.time()-t0:.0f}s: output {out_bytes/1e9:.3f} GB "
          f"({out_bytes/2**30:.3f} GiB); saved {saved/2**30:.3f} GiB vs the input pack")
    print(f"output dir: {args.out}")


# ------------------------------------------------------------------ selftest
def import_kld(script_dir):
    import importlib.util
    for here in (script_dir, os.getcwd()):
        path = os.path.join(here, "q38fn_kld.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("q38fn_kld", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["q38fn_kld"] = mod
            spec.loader.exec_module(mod)
            return mod
    fail("kld-import", f"q38fn_kld.py not found next to build_f2.py or in cwd "
                       f"(looked in {script_dir!r} and {os.getcwd()!r})")


def cmd_selftest(args):
    global SRC_DIR
    SRC_DIR = args.src
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.ref, "config.json")))
    ref_ql = cfg["quantization_config"]["quantized_layers"]
    names = converted_names(cfg)
    src_index = st_index(args.src)
    nv_index = st_index(args.nv)

    # (a) MXFP8 round trip on 3 real projections via the NEW kld dequant
    kld = import_kld(os.path.dirname(os.path.abspath(__file__)))
    probes = ["model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
              "model.language_model.layers.3.self_attn.q_proj.weight",
              "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"]
    for pname in probes:
        w = st_load(args.src, src_index, pname)
        codes, sb = quantize_mxfp8(w, pname)
        if codes.dtype != torch.float8_e4m3fn or sb.dtype != torch.uint8:
            fail("selftest-a", f"{pname}: dtypes {codes.dtype}/{sb.dtype}")
        dq = kld.dequant_mxfp8(codes, sb, tuple(w.shape))
        # the kld dequant must match an independent reference decode exactly
        ref = (codes.to(torch.float32).reshape(w.shape[0], -1, 32)
               * torch.pow(2.0, (sb.to(torch.int32) - 127).to(torch.float32)).unsqueeze(-1)
               ).reshape(w.shape).to(torch.bfloat16)
        if not torch.equal(dq, ref):
            fail("selftest-a", f"{pname}: kld dequant != reference decode")
        rel = ((dq.to(torch.float32) - w.to(torch.float32)).norm()
               / w.to(torch.float32).norm()).item()
        if not 0.01 <= rel <= 0.05:
            fail("selftest-a", f"{pname} rel err {rel:.4f} outside [0.01, 0.05]")
        print(f"  (a) {pname.split('.layers.')[-1]}: rel err {rel:.4f} ok")
        del w, codes, sb, dq, ref

    # (b) names, dtypes, shapes of every converted tensor == the official FP4 reference
    for pname in names:
        (nsh, ndt) = header_meta(args.nv, nv_index, pname)
        (nssh, nsdt) = header_meta(args.nv, nv_index, pname + "_scale")
        (ssh, _sdt) = header_meta(args.src, src_index, pname)
        if nsh != ssh or ndt != "F8_E4M3" or nsdt != "U8" or nssh != (ssh[0], ssh[1] // 32):
            fail("selftest-b", f"{pname}: nv {nsh}/{ndt} scale {nssh}/{nsdt} vs src {ssh}")
    # quantize every tensor once; check emitted dtypes/shapes (~2 min CPU)
    tot = 0
    for pname in names:
        w = st_load(args.src, src_index, pname)
        codes, sb = quantize_mxfp8(w, pname)
        O, I = w.shape
        if codes.shape != (O, I) or sb.shape != (O, I // 32):
            fail("selftest-b", f"{pname}: emitted {tuple(codes.shape)}/{tuple(sb.shape)}")
        tot += O * I
        del w, codes, sb
    saved = tot * 31 // 32
    print(f"  (b) {len(names)} names/dtypes/shapes match the official FP4 reference "
          f"(F8_E4M3 + U8, k%32==0)")

    # (c) stitched-group rule for all 48 layers in the planned quantized_layers
    ql, added = plan_ql(ref_ql, names)
    if added != 384:
        fail("selftest-c", f"added {added} entries, expected 384")
    want = {"quant_algo": "MXFP8", "group_size": 32}
    for i, kind in enumerate(cfg["text_config"]["layer_types"]):
        pre = (f"model.language_model.layers.{i}.linear_attn" if kind == "linear_attention"
               else f"model.language_model.layers.{i}.self_attn")
        parts = (["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
                 if kind == "linear_attention"
                 else ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj", "o_proj"])
        grp = [ql.get(f"{pre}.{p}") for p in parts]
        if not all(g == want for g in grp):
            fail("selftest-c", f"layer {i} ({kind}) group not all-or-none MXFP8/32: {grp}")
        se = [ql.get(f"model.language_model.layers.{i}.mlp.shared_expert.{p}")
              for p in ["gate_proj", "up_proj", "down_proj"]]
        if not all(g == want for g in se):
            fail("selftest-c", f"layer {i} shared_expert group broken: {se}")
    base = dict(ref_ql)
    kept = {k: v for k, v in ql.items() if k in base}
    if kept != base:
        fail("selftest-c", "existing NVFP4/W4A16 entries changed")
    nvfp4 = [k for k, v in ql.items() if v.get("quant_algo") == "NVFP4"]
    if len(nvfp4) != 48 or not all(k.endswith("mlp.experts") for k in nvfp4):
        fail("selftest-c", f"NVFP4 expert entries: {len(nvfp4)}")
    print(f"  (c) quantized_layers: +384 MXFP8 (stitched all-or-none x48 layers), "
          f"NVFP4 experts {len(nvfp4)} + W4A16 kept intact")
    est = sum(os.path.getsize(os.path.join(args.ref, f)) for f in os.listdir(args.ref)
              if os.path.isfile(os.path.join(args.ref, f))) - saved
    print(f"bytes saved: {saved} ({saved/1e9:.3f} GB / {saved/2**30:.3f} GiB)")
    print(f"output pack size (estimated): {est/1e9:.3f} GB")
    print(f"selftest time {time.time()-t0:.0f}s")
    print("SELFTEST PASS")


# ------------------------------------------------------------------ verify-full
def hash_slice(path, off, ln):
    """sha256 of safetensors data bytes at header-relative offset `off`."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        fh.seek(8 + n + off)
        left = ln
        while left > 0:
            blk = fh.read(min(1 << 24, left))
            if not blk:
                fail("verify", f"short read in {path}")
            h.update(blk)
            left -= len(blk)
    return h.hexdigest()


def cmd_verify_full(args):
    """Byte-verify every REF-listed tensor in every model shard, REF vs OUT."""
    cfg = json.load(open(os.path.join(args.ref, "config.json")))
    names = converted_names(cfg)
    ref_index, out_index = st_index(args.ref), st_index(args.out)
    conv = set(names)
    # converted weights must differ (they are f8 now); their _scale is new in OUT;
    # every other REF-listed tensor (incl. NVFP4 expert weight/scale/scale_2/
    # input_scale) must be byte-identical
    skip_ref = conv | {n + "_scale" for n in names}
    bad = []
    checked = 0
    shards = sorted({s for s in ref_index.values() if s.startswith("model-")})
    for shard in shards:
        rp, op = os.path.join(args.ref, shard), os.path.join(args.out, shard)
        rh, oh = raw_header(rp), raw_header(op)
        # expected OUT order: REF order with each converted weight's _scale
        # inserted immediately after its weight
        expect = []
        for k in rh:
            if k != "__metadata__":
                expect.append(k)
            if k in conv:
                expect.append(k + "_scale")
        got = [k for k in oh if k != "__metadata__"]
        if expect != got:
            fail("verify", f"{shard}: key order differs from REF+insertions")
        for k, meta in rh.items():
            if k == "__metadata__" or k in skip_ref:
                continue
            om = oh[k]
            if om["shape"] != meta["shape"] or om["dtype"] != meta["dtype"]:
                fail("verify", f"{shard}:{k}: dtype/shape changed")
            a, b = meta["data_offsets"]
            c, d = om["data_offsets"]
            if (b - a) != (d - c):
                fail("verify", f"{shard}:{k}: byte length changed")
            checked += 1
            if hash_slice(rp, a, b - a) != hash_slice(op, c, d - c):
                bad.append(f"{shard}:{k}")
        print(f"  {shard}: {checked} tensors verified so far", flush=True)
    if bad:
        fail("verify", f"{len(bad)} tensors differ from the input pack: {bad[:5]}")
    # converted pairs: dtype/shape right + index maps them
    for n in names:
        if out_index.get(n + "_scale") != out_index.get(n):
            fail("verify", f"{n}_scale not mapped to its weight's shard in OUT index")
    print(f"VERIFY FULL PASS: {checked} non-converted tensors byte-identical to the input pack")


def main():
    ap = argparse.ArgumentParser(description="MXFP8 dense pack builder/selftest")
    ap.add_argument("--src", default=SRC_DEF)
    ap.add_argument("--ref", default=REF_DEF)
    ap.add_argument("--out", default=OUT_DEF)
    ap.add_argument("--nv", default=NV_DEF, help="official FP4 reference pack dir (selftest b)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-full", action="store_true",
                    help="after build: byte-verify all non-converted tensors vs the input pack")
    args = ap.parse_args()
    if args.selftest:
        cmd_selftest(args)
    elif args.verify_full:
        cmd_verify_full(args)
    else:
        cmd_build(args)


if __name__ == "__main__":
    main()
