#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1b: speed-of-light (SOL / roofline) calculator.
#
# Two uses:
#   1. --rgp CAPTURE: an rgp_capture.ps1 capture (its *_dispatches.csv); every kernel class gets the bytes / FLOPs it
#      must move / compute in that capture (from the real tensor shapes in sol_shapes.json), its achieved rate, the
#      hardware ceiling, and a priority = runtime share x headroom (1 - achieved / ceiling). Ranked list.
#   2. --theory: no capture; per op class the bytes and FLOPs of one decode step, one MTP verify forward (n rows)
#      or one prefill chunk at a given depth, and the time at the ceiling (SOL time) on either GPU.
#
#   python sol.py --rgp fl1a\rgp\20261015-085640_8s_balance_ctx2048_decode --device 8s [--model dense] [--ctx 4096]
#   python sol.py --theory --model dense|moe --device 8s|r97 --phase decode|verify|prefill --ctx N [--rows 4|2048]
#   python sol.py --fl1a        (the four FL-1a 8060S captures, the FL-1b audit tables)
#
# Byte / FLOP model (per class; the engine facts it relies on are cited at each line in code):
#   decode GEMV of type T : every weight of type T is read once per forward (int8 GEMV, weights dominate); the
#                           capture window runs from the first to the last LM-head launch = whole verify cycles,
#                           each = one main forward (n verify rows) + its MTP draft steps (MTP layer + 64k-row
#                           2-bit draft head). Dense MTP q/o/gate/up/down/eh_proj are requantized to q4_k at load
#                           (loader.cpp requantMtpQ4).
#   LM head               : output.weight once per verify forward.
#   decode attention      : K+V of every cached token read once per attention layer and forward (q8h: 1 byte +
#                           an f16 scale per 32 values = 2176 B / token / layer for 4 KV heads x 256).
#   GDN decode step       : the f32 recurrent state [v_heads][dv][dk] read + written once per launch.
#   prefill GEMM          : 2 * rows * N * K per weight (WMMA f16; dequant to f16 tiles first); compute-bound.
#   prefill dequant       : quant bytes read + f16 written per weight.
#   prefill attention     : 4 * n_head * head_dim * rows * (depth + rows / 2) (QK^T + PV, causal in the chunk).
#   GDN chunk             : ROUGH: per token and v-head 6*C*dk + 4*C*dv (intra-chunk: KK^T, W, U, QK^T, AV; C = 64)
#                           + 6*dk*dv (QS, WS, K^T U) FLOPs (chunked gated delta rule, FLA-style); f32/f16 mix.
#   other (norms, quantize, copies, argmax ...): time only, no ceiling (counted in the share, priority n/a).
# RGP raises the clocks to peak during a capture (driver: 'Successfully set clocks to peak'): read the achieved %
# and the ratios, not the absolute times.
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = r"C:\ClaudeWorkspace\whirl-work"

# Hardware ceilings. 'peak' = theoretical / measured register peak; 'prac' = the best measured on this machine.
HW = {
    "8s": dict(name="Radeon 8060S (gfx1151)",
               bw=256e9,      # LPDDR5X-8000 x 256 bit
               bw_prac=238e9,  # hipMalloc copy measured (audit/KGAP-1.md); rocm_bandwidth_test 212 GB/s (llm-tracker)
               f16=59.4e12,   # 40 CU x 2.9 GHz x 512 FLOP/clk (llm-tracker Strix Halo page)
               f16_prac=25.0e12,  # hipBLASLt 1.2.2 on this box, WHIRL shapes: 19.5-25.0 TFLOPS (audit/PEAK-1.md)
               f16_rgp=22e12,  # practical f16 ceiling (KLIB-1 2026-10-15): hipBLASLt sustains 20-25, strix/llama.cpp FA 18-21 TF/s even at RGP clocks; FAST-0 register peak 43.7 overstated headroom ~2x
               iu8=59.4e12),  # iu8 = f16 speed on RDNA3.5 (PEAK-1: hipBLASLt int8 18-34 TOPS = f16)
    "r97": dict(name="Radeon AI PRO R9700 (gfx1201)",
                bw=640e9,       # 20 Gbps GDDR6 x 256 bit
                bw_prac=618e9,  # 610-626 GB/s measured (audit/KGAP-1.md)
                f16=189e12,     # WMMA f16 measured register peak (FAST-0, audit/LNX-1.md)
                f16_prac=130e12,  # hipBLASLt 116-129 / gemmh_f16 126-130 TFLOPS (audit/PEAK-1.md, LNX-1.md)
                f16_rgp=130e12,  # R9700 is not power-limited in GEMM: same as f16_prac
                iu8=377e12),    # WMMA iu8 measured (FAST-0)
}

# ggml block sizes: (values per block, bytes per block)
QB = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22),
      "Q5_1": (32, 24), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
      "IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "IQ3_S": (256, 110), "IQ3_XXS": (256, 98), "MXFP4": (32, 17)}
KTYPES = ["q4_k", "q5_k", "q6_k", "q3_k", "q8_0", "iq4_xs", "iq4_nl", "iq3_s", "mxfp4", "q2_k", "iq3_xxs", "f16", "f32"]
KV_BYTES = {"q8h": 2176, "q4": 1152, "f16": 4096}  # per token per attention layer, 4 KV heads x 256 (K + V)


def load_shapes(model):
    sh = json.load(open(os.path.join(HERE, "sol_shapes.json"), encoding="utf-8"))[model]
    meta = sh["meta"]
    n_all = int(meta["block_count"])
    n_next = int(meta.get("nextn_predict_layers", 0))
    m = dict(arch=sh["arch"], n_layer=n_all - n_next, n_next=n_next, E=int(meta["embedding_length"]),
             n_head=int(meta["attention.head_count"]), n_kv=int(meta["attention.head_count_kv"]), hd=int(meta["attention.key_length"]),
             interval=int(meta.get("full_attention_interval", 4)), v_heads=int(meta["ssm.time_step_rank"]),
             dk=int(meta["ssm.state_size"]), dv=int(meta["ssm.inner_size"]) // int(meta["ssm.time_step_rank"]),
             n_expert=int(meta.get("expert_count", 0)), n_used=int(meta.get("expert_used_count", 0)))
    m["tensors"] = [dict(name=n, type=t, shape=s, bytes=b) for n, t, s, b in sh["tensors"]]
    return m


def is_attn(m, il):
    return (il + 1) % m["interval"] == 0  # every interval-th layer is full attention (layers 3, 7, ...)


def mats(m, layers, mtp=False):
    """Matmul weights (2-D/3-D, not conv / norms) of the given layer indices: list of dicts with type, N, K, bytes,
    params, experts (3-D)."""
    out = []
    for t in m["tensors"]:
        mm = re.match(r"blk\.(\d+)\.(.+)", t["name"])
        if not mm or int(mm.group(1)) not in layers:
            continue
        s = t["shape"]
        if len(s) < 2 or "conv1d" in t["name"] or s[1] == 1:
            continue
        ty = t["type"]
        if mtp and m["arch"] == "qwen35" and ty in ("Q6_K", "Q8_0") and any(k in t["name"] for k in ("attn_q.", "attn_output", "ffn_gate.", "ffn_up.", "ffn_down.", "eh_proj")):
            ty = "Q4_K"  # requantMtpQ4 (loader.cpp): dense MTP q / o / gate / up / down / eh_proj -> q4_k
        vb, bb = QB[ty]
        ne = s[2] if len(s) > 2 else 1
        out.append(dict(name=t["name"], type=ty.lower(), K=s[0], N=s[1], experts=ne, params=s[0] * s[1] * ne, bytes=s[0] * s[1] * ne // vb * bb))
    return out


def head_bytes(m):
    t = [x for x in m["tensors"] if x["name"] == "output.weight"] or [x for x in m["tensors"] if x["name"] == "token_embd.weight"]
    return t[0]["bytes"], t[0]["type"].lower(), t[0]["shape"]


def expert_frac(m, rows):
    """Expected fraction of experts touched by `rows` tokens with top-k routing (uniform, independent: an estimate)."""
    if not m["n_expert"]:
        return 1.0
    return 1.0 - (1.0 - m["n_used"] / m["n_expert"]) ** rows


def eff(m, w, rows):
    """(bytes read, params that do FLOPs per row) of weight w for `rows` tokens: an expert tensor is read for the experts
    the rows touch (expert_frac) but each row multiplies only its n_used experts."""
    if w["experts"] > 1:
        return w["bytes"] * expert_frac(m, rows), w["params"] * m["n_used"] / m["n_expert"]
    return w["bytes"], w["params"]


def weight_bytes_by_type(m, layers, rows, mtp=False):
    d = {}
    for w in mats(m, layers, mtp):
        b = w["bytes"] * (expert_frac(m, rows) if w["experts"] > 1 else 1.0)
        d[w["type"]] = d.get(w["type"], 0.0) + b
    return d


def ktype(kernel):
    for t in KTYPES:
        if kernel.endswith("_" + t):
            return t
    return None


# ----------------------------------------------------------------------------------------------------------------------
# --rgp: classify the capture's dispatches, attach bytes / FLOPs, rank


def classify(k):
    if k.startswith(("gemm_", "gemms_", "gemmh", "gemm8")):
        return "prefill GEMM"
    if k.startswith("dequant_f16_"):
        return "prefill dequant " + (ktype(k) or "?")
    if k.startswith("attn_prefill"):
        return "prefill attention"
    if k.startswith("gdn_chunk") or k.startswith("gdn_conv"):
        return "GDN chunk (prep/scan/conv)"
    if k.startswith("gdn_step"):
        return "GDN decode step"
    if k.startswith("gemv_d2"):
        return "draft head (2-bit 64k)"
    if re.match(r"g?gemv", k):
        return "decode GEMV " + (ktype(k) or "?")
    if k.startswith(("attn_ws", "attn_combine", "attn_prep", "attn_kg", "attn_dec", "attn_wsplit")):
        return "decode attention (+prep/combine)"
    return "other"


def rgp_analyze(prefix, dev, model, ctx, kv, rows_prefill):
    disp = prefix if prefix.endswith("_dispatches.csv") else prefix + "_dispatches.csv"
    rows = [d for d in csv.DictReader(open(disp, encoding="utf-8"))]
    for d in rows:
        d["idx"], d["dur"], d["threads"] = int(d["idx"]), float(d["dur_us"]), int(d["threads"] or 0)
    m = load_shapes(model)
    hw = HW[dev]
    phase = "decode" if any(re.match(r"g?gemvq", d["kernel"]) for d in rows) and not any(d["kernel"].startswith("gemm_") for d in rows) else "prefill"
    if ctx is None:
        mm = re.search(r"ctx(\d+)", os.path.basename(disp))
        ctx = int(mm.group(1)) if mm else 4096
        ctx += 2048 if phase == "decode" else 0  # decode: the capture's request adds ~2k tokens to the KV first
    res = {}
    total = sum(d["dur"] for d in rows)
    notes = []
    if phase == "decode":
        hb, ht, hs = head_bytes(m)
        V = hs[1]
        cand = [d for d in rows if re.match(r"g?gemv", d["kernel"]) and not d["kernel"].startswith("gemv_d2")]
        hmax = max(d["threads"] for d in cand)
        heads = [d["idx"] for d in cand if d["threads"] == hmax]
        if len(heads) < 2:
            raise SystemExit("sol: fewer than 2 LM-head launches in the capture: no whole verify cycle")
        lo, hi = heads[0], heads[-1]
        win = [d for d in rows if lo < d["idx"] <= hi]
        cycles = len(heads) - 1
        drafts = sum(1 for d in win if d["kernel"].startswith("gemv_d2"))
        n_attn = sum(1 for il in range(m["n_layer"]) if is_attn(m, il))
        n_gdn = m["n_layer"] - n_attn
        gdn_launch = sum(1 for d in win if d["kernel"].startswith("gdn_step"))
        attn_launch = sum(1 for d in win if d["kernel"].startswith("attn_ws"))
        notes.append(f"window: launches {lo + 1}..{hi} = {cycles} verify cycles + {drafts} draft steps "
                     f"(GDN step launches {gdn_launch} vs {cycles}x{n_gdn} = {cycles * n_gdn}; attention launches {attn_launch} vs "
                     f"{cycles}x{n_attn} + {drafts} = {cycles * n_attn + drafts})")
        main = weight_bytes_by_type(m, set(range(m["n_layer"])), 4)
        mtp = weight_bytes_by_type(m, set(range(m["n_layer"], m["n_layer"] + m["n_next"])), 1, mtp=True)
        by = {t: cycles * main.get(t, 0) + drafts * mtp.get(t, 0) for t in set(main) | set(mtp)}
        d2_bytes = 65536 * (m["E"] // 4 + m["E"] // 16)  # requant_q6k_d2: 2-bit codes + f16 scale per 16; 64k draft vocab
        kvb = KV_BYTES[kv]
        for d in win:
            c = classify(d["kernel"])
            if d["threads"] == hmax and d["kernel"] in {x["kernel"] for x in cand}:
                c = "LM head (" + ht.lower() + ")"
            r = res.setdefault(c, dict(us=0.0, n=0, bytes=None, flops=None))
            r["us"] += d["dur"]
            r["n"] += 1
        for c, r in res.items():
            if c.startswith("decode GEMV "):
                r["bytes"] = by.get(c.split()[-1], 0) or None
                r["bound"] = "bw"
            elif c.startswith("LM head"):
                r["bytes"], r["bound"] = cycles * hb, "bw"
            elif c.startswith("draft head"):
                r["bytes"], r["bound"] = drafts * d2_bytes, "bw"
            elif c.startswith("decode attention"):
                r["bytes"], r["bound"] = (cycles * n_attn + drafts) * ctx * kvb, "bw"
            elif c == "GDN decode step":
                r["bytes"], r["bound"] = r["n"] * 2 * m["v_heads"] * m["dv"] * m["dk"] * 4, "bw"
        total = sum(d["dur"] for d in win)
        # row coverage check of the GEMV byte model: launched rows (WGs x rows per WG) vs expected
        exp_rows = cycles * sum(w["N"] for w in mats(m, set(range(m["n_layer"])))) + drafts * sum(w["N"] for w in mats(m, set(range(m["n_layer"], m["n_layer"] + m["n_next"]))))
        got_rows = 0
        for d in win:
            k = d["kernel"]
            if re.match(r"g?gemvq", k) and d["threads"] != hmax:
                rpb = 16 if re.search(r"nt\d+r2_", k) else 8  # gemvKernels: 8 rows per block, x gemv_r (r2) for the mr kernels
                got_rows += d["threads"] // 256 * rpb
        notes.append(f"GEMV row check: launched rows {got_rows} vs expected {exp_rows} ({100.0 * got_rows / exp_rows:.1f}%); "
                     "differences = rows padded to whole blocks, the small router / alpha / beta GEMVs, MTP rows")
    else:
        R = rows_prefill
        n_gdn_cov = sum(1 for d in rows if d["kernel"] == "gdn_chunk_scan")
        n_attn_cov = sum(1 for d in rows if d["kernel"].startswith("attn_prefill"))
        depth = (ctx if ctx else 0)
        gl = [il for il in range(m["n_layer"]) if not is_attn(m, il)]
        al = [il for il in range(m["n_layer"]) if is_attn(m, il)]
        pg = sum(eff(m, w, R)[1] for w in mats(m, set(gl))) / len(gl)
        pa = sum(eff(m, w, R)[1] for w in mats(m, set(al))) / len(al)
        bg = sum(w["bytes"] for w in mats(m, set(gl))) / len(gl)
        ba = sum(w["bytes"] for w in mats(m, set(al))) / len(al)
        params = n_gdn_cov * pg + n_attn_cov * pa
        notes.append(f"capture covers {n_gdn_cov} GDN + {n_attn_cov} attention layers of {m['n_layer']} (SQTT buffer); "
                     f"chunk {R} rows at depth {depth}")
        for d in rows:
            c = classify(d["kernel"])
            r = res.setdefault(c, dict(us=0.0, n=0, bytes=None, flops=None))
            r["us"] += d["dur"]
            r["n"] += 1
        for c, r in res.items():
            if c == "prefill GEMM":
                r["flops"], r["bound"] = 2.0 * R * params, "f16"
            elif c.startswith("prefill dequant"):
                pass  # set below (by type)
            elif c == "prefill attention":
                r["flops"], r["bound"] = n_attn_cov * 4.0 * m["n_head"] * m["hd"] * R * (depth + R / 2), "f16"
            elif c.startswith("GDN chunk"):
                r["flops"], r["bound"] = n_gdn_cov * R * m["v_heads"] * (6 * 64 * m["dk"] + 4 * 64 * m["dv"] + 6 * m["dk"] * m["dv"]), "f16"
                r["rough"] = True
        # dequant: quant bytes in + f16 out of the covered layers, by type
        cov = set(gl[:n_gdn_cov]) | set(al[:n_attn_cov])
        dq = {}
        for w in mats(m, cov):
            dq[w["type"]] = dq.get(w["type"], 0) + w["bytes"] + 2 * w["params"]
        for c, r in res.items():
            if c.startswith("prefill dequant"):
                r["bytes"], r["bound"] = dq.get(c.split()[-1], None), "bw"
    out = []
    for c, r in res.items():
        share = r["us"] / total
        ach = ceil = ceilp = None
        if r.get("bytes") and r.get("bound") == "bw":
            ach = r["bytes"] / (r["us"] * 1e-6)
            ceil, ceilp = hw["bw"], hw["bw_prac"]
        elif r.get("flops") and r.get("bound") == "f16":
            ach = r["flops"] / (r["us"] * 1e-6)
            ceil, ceilp = hw["f16"], hw["f16_rgp"]
        frac = ach / ceil if ach else None
        fracp = ach / ceilp if ach else None
        prio = share * max(0.0, 1.0 - min(1.0, fracp)) if fracp is not None else None
        # > 105% of peak: the byte model is too high for this class (small tensors served from cache, or not every
        # tensor of the type takes this kernel path): flagged, no priority
        over = frac is not None and frac > 1.05
        if over:
            prio = None
        out.append(dict(cls=c, us=r["us"], n=r["n"], share=share, ach=ach, frac=frac, fracp=fracp, prio=prio, bound=r.get("bound"),
                        rough=r.get("rough", False), over=over))
    out.sort(key=lambda x: -(x["prio"] if x["prio"] is not None else -1 + x["share"] * 1e-3))
    return dict(phase=phase, ctx=ctx, total_us=total, rows=out, notes=notes, hw=hw, model=model, prefix=prefix)


def fmt_rate(x, bound):
    if x is None:
        return "-"
    return f"{x / 1e9:.0f} GB/s" if bound == "bw" else f"{x / 1e12:.1f} TF/s"


def print_table(a, md=False):
    hw = a["hw"]
    print(f"\n== SOL {os.path.basename(a['prefix'])}: {a['model']} {a['phase']} ctx {a['ctx']} on {hw['name']} "
          f"(ceilings: {hw['bw'] / 1e9:.0f} GB/s peak / {hw['bw_prac'] / 1e9:.0f} measured; f16 {hw['f16'] / 1e12:.1f} peak / "
          f"{hw['f16_rgp'] / 1e12:.1f} at RGP peak clock)")
    for n in a["notes"]:
        print("   " + n)
    if md:
        print("\n| rank | class | launches | time share | achieved | % of peak | % of measured ceiling | priority = share x headroom |")
        print("|---|---|---:|---:|---:|---:|---:|---:|")
    else:
        print(f"{'rank':>4} {'class':38s} {'n':>5} {'share':>7} {'achieved':>11} {'%peak':>6} {'%meas':>6} {'prio':>6}")
    for i, r in enumerate(a["rows"], 1):
        pk = f"{100 * r['frac']:.0f}%" if r["frac"] is not None else "-"
        pm = f"{100 * r['fracp']:.0f}%" if r["fracp"] is not None else "-"
        pr = f"{100 * r['prio']:.1f}" if r["prio"] is not None else "n/a"
        name = r["cls"] + (" (rough)" if r["rough"] else "") + (" (>peak: cache/path)" if r.get("over") else "")
        if md:
            print(f"| {i} | {name} | {r['n']} | {100 * r['share']:.1f}% | {fmt_rate(r['ach'], r['bound'])} | {pk} | {pm} | {pr} |")
        else:
            print(f"{i:>4} {name:38s} {r['n']:>5} {100 * r['share']:6.1f}% {fmt_rate(r['ach'], r['bound']):>11} {pk:>6} {pm:>6} {pr:>6}")
    print("   priority = time share x (1 - achieved / measured ceiling), in points of the captured GPU time; RGP runs at "
          "peak clocks, so compare ratios, not absolute times. f16 'measured' under RGP = the unthrottled register peak "
          "(8060S FAST-0 43.7 TF/s); real GEMMs top out near 67-69% of the register peak (R9700 hipBLASLt / gemmh_f16).")


# ----------------------------------------------------------------------------------------------------------------------
# --theory


def theory(model, dev, phase, ctx, rows, kv):
    m = load_shapes(model)
    hw = HW[dev]
    L = set(range(m["n_layer"]))
    n_attn = sum(1 for il in L if is_attn(m, il))
    n_gdn = m["n_layer"] - n_attn
    hb, ht, hs = head_bytes(m)
    ops = []
    if phase in ("decode", "verify"):
        n = 1 if phase == "decode" else rows
        for t, b in sorted(weight_bytes_by_type(m, L, n).items(), key=lambda x: -x[1]):
            fl = 2.0 * n * sum(eff(m, w, n)[1] for w in mats(m, L) if w["type"] == t)
            ops.append((f"GEMV {t}", b, fl))
        ops.append((f"LM head ({ht.lower()})", hb, 2.0 * n * hs[0] * hs[1]))
        kvb = n_attn * ctx * KV_BYTES[kv]
        ops.append((f"attention KV ({kv}, {n_attn} layers)", kvb, 4.0 * n * m["n_head"] * m["hd"] * ctx * n_attn))
        st = n_gdn * 2 * m["v_heads"] * m["dv"] * m["dk"] * 4
        ops.append((f"GDN state r+w ({n_gdn} layers, f32)", st, n * n_gdn * m["v_heads"] * 6 * m["dk"] * m["dv"]))
    else:
        R, D = rows, ctx
        for t, b in sorted(weight_bytes_by_type(m, L, R).items(), key=lambda x: -x[1]):
            ws = [w for w in mats(m, L) if w["type"] == t]
            ps = sum(eff(m, w, R)[1] for w in ws)
            # dense weights: dequant to f16 tiles (written + read back: 4 B / value; the 8060S path); expert weights:
            # dequant fused in the grouped MoE GEMM (quant bytes only)
            f16b = sum(4 * w["params"] for w in ws if w["experts"] == 1)
            ops.append((f"GEMM {t}", b + f16b, 2.0 * R * ps))
        ops.append(("attention prefill", R * n_attn * KV_BYTES[kv] + D * n_attn * KV_BYTES[kv],
                    n_attn * 4.0 * m["n_head"] * m["hd"] * R * (D + R / 2)))
        ops.append(("GDN chunk (rough)", n_gdn * 2 * m["v_heads"] * m["dv"] * m["dk"] * 4,
                    n_gdn * R * m["v_heads"] * (6 * 64 * m["dk"] + 4 * 64 * m["dv"] + 6 * m["dk"] * m["dv"])))
        ops.append((f"LM head (last row only)", hb, 2.0 * hs[0] * hs[1]))
    print(f"\n== SOL theory: {model} {phase} ctx {ctx}{' rows ' + str(rows) if phase != 'decode' else ''} on {hw['name']}")
    print(f"{'op':40s} {'MB':>9} {'GFLOP':>9} {'t_bw ms':>8} {'t_f16 ms':>9} {'bound':>6} {'SOL ms':>8} {'(meas. ceil)':>12}")
    tot = totp = 0.0
    for name, b, fl in ops:
        tb, tf = b / hw["bw"] * 1e3, fl / hw["f16"] * 1e3
        tbp, tfp = b / hw["bw_prac"] * 1e3, fl / hw["f16_prac"] * 1e3
        s, sp = max(tb, tf), max(tbp, tfp)
        tot += s
        totp += sp
        print(f"{name:40s} {b / 1e6:9.1f} {fl / 1e9:9.1f} {tb:8.2f} {tf:9.2f} {'bw' if tb >= tf else 'f16':>6} {s:8.2f} {sp:12.2f}")
    print(f"{'total (no overlap between ops)':40s} {'':9s} {'':9s} {'':8s} {'':9s} {'':6s} {tot:8.2f} {totp:12.2f}")
    if phase == "decode":
        print(f"SOL plain decode: {1e3 / tot:.1f} tok/s at peak, {1e3 / totp:.1f} tok/s at the measured ceilings (no MTP)")
    elif phase == "prefill":
        print(f"SOL prefill: {rows / tot * 1e3:.0f} tok/s at peak, {rows / totp * 1e3:.0f} tok/s at the measured ceilings (chunk at depth {ctx})")
    return ops


FL1A = [("20261015-085640_8s_balance_ctx2048_decode", "decode @4k"), ("20261015-100306_8s_balance_ctx126976_decode", "decode @131k"),
        ("20261015-085152_8s_balance_ctx2048_prefill", "prefill 2k chunk @2k"), ("20261015-100306_8s_balance_ctx126976_prefill", "prefill 2k chunk @127k")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgp", help="capture prefix or *_dispatches.csv from rgp_capture")
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--fl1a", action="store_true", help="analyze the four FL-1a 8060S captures")
    ap.add_argument("--device", choices=["8s", "r97"], default="8s")
    ap.add_argument("--model", choices=["dense", "moe"], default="dense")
    ap.add_argument("--phase", choices=["decode", "verify", "prefill"], default="decode")
    ap.add_argument("--ctx", type=int, default=None, help="KV depth (decode) / chunk start depth (prefill)")
    ap.add_argument("--rows", type=int, default=None, help="verify rows (default 4) / prefill chunk rows (default 2048)")
    ap.add_argument("--kv", choices=list(KV_BYTES), default="q8h")
    ap.add_argument("--md", action="store_true", help="markdown tables")
    a = ap.parse_args()
    if a.fl1a:
        for p, label in FL1A:
            path = os.path.join(WORK, "fl1a", "rgp", p)
            ctx = None
            if "prefill" in p:
                # the prefill capture = request B: the ~2k rows after the request-A checkpoint at Ctx (2042 / 126976)
                ctx = 2042 if "ctx2048" in p else 126976
            print(f"\n### {label}")
            print_table(rgp_analyze(path, "8s", "dense", ctx, a.kv, 2052 if "ctx2048" in p else 2048), md=a.md)
        return 0
    if a.rgp:
        rows = a.rows or 2048
        print_table(rgp_analyze(a.rgp, a.device, a.model, a.ctx, a.kv, rows), md=a.md)
        return 0
    if a.theory:
        rows = a.rows or (4 if a.phase == "verify" else 2048)
        theory(a.model, a.device, a.phase, a.ctx if a.ctx is not None else 4096, rows, a.kv)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
