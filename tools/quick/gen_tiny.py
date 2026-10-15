#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1b: tiny random-weight test models (qwen35 dense / qwen35moe) for server, scheduler, cache and tier tests.
#
# A GGUF with the real per-layer shapes, head counts, quant types, tokenizer and chat template of a real model, but
# only --layers main layers (default 2: one DeltaNet + one full-attention layer, full_attention_interval 2) plus its
# MTP (nextn) layer. Weights are random: every tensor is the matching real tensor with its quant BLOCKS randomly
# permuted (f32 vectors: elements permuted), seeded per tensor name. So every block is a valid encoding of its
# type with realistic scales (no NaN / overflow), every kernel path of the real model runs, and greedy output is
# deterministic, but the weights carry no structure of the original model (the text is noise).
# llama.cpp makes its per-arch test models the same way in spirit (tests/test-llama-archs.cpp: tiny n_embd 256,
# 2 layers, full_attention_interval 2, normal-random weights); WHIRL needs the full tokenizer and the real
# shapes its kernels are tuned for, hence real shapes + permuted real blocks here.
# Reads / writes GGUF with llama.cpp's gguf-py (MIT, C:\ClaudeWorkspace\refsrc\llama.cpp\gguf-py; see
# THIRD_PARTY_NOTICES.md "llama.cpp / ggml" for the MIT license; imported at run time, not copied); developer tool,
# not shipped.
#
#   python gen_tiny.py --model dense|moe [--layers 2] [--interval N] [--no-mtp] [--weights shuffle|copy] [--seed 1]
#                      [--out C:\ClaudeWorkspace\whirl-work\fl1b\tiny\NAME.gguf]
import argparse
import hashlib
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\ClaudeWorkspace\refsrc\llama.cpp\gguf-py")
from gguf import GGUFReader, GGUFValueType, GGUFWriter, GGMLQuantizationType  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

SRC = {"dense": r"C:\Users\tsaip\.lmstudio\models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_M.gguf",
       "moe": r"C:\Users\tsaip\.lmstudio\models\ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf"}
OUTDIR = r"C:\ClaudeWorkspace\whirl-work\fl1b\tiny"


def field_value(f):
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        st = f.types[1] if len(f.types) > 1 else None
        if st == GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode("utf-8", errors="surrogateescape") for i in f.data], t, st
        return [f.parts[i].tolist()[0] for i in f.data], t, st
    if t == GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode("utf-8", errors="surrogateescape"), t, None
    return f.parts[f.data[0]].tolist()[0], t, None


def rng_for(seed, name):
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def randomize(t, seed, mode):
    """t: GGUFReader tensor -> (array to write, raw_dtype or None)."""
    data = np.asarray(t.data)
    qt = t.tensor_type
    if mode == "copy":
        return data.copy(), (qt if qt not in (GGMLQuantizationType.F32, GGMLQuantizationType.F16) else None)
    r = rng_for(seed, t.name)
    if qt in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        flat = data.reshape(-1).copy()
        r.shuffle(flat)
        return flat.reshape(data.shape), None
    bs, bb = GGML_QUANT_SIZES[qt]
    raw = data.reshape(-1).view(np.uint8)
    blocks = raw.reshape(-1, bb).copy()
    r.shuffle(blocks, axis=0)
    return blocks.reshape(data.shape), qt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SRC), default="dense")
    ap.add_argument("--src", default="")
    ap.add_argument("--layers", type=int, default=2, help="main layers (default 2)")
    ap.add_argument("--interval", type=int, default=0, help="full_attention_interval (default: 2 if layers < 4, else the source's)")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--weights", choices=["shuffle", "copy"], default="shuffle")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    src = a.src or SRC[a.model]
    t0 = time.time()
    rd = GGUFReader(src)
    arch = field_value(rd.fields["general.architecture"])[0]
    K = lambda s: f"{arch}.{s}"  # noqa: E731
    n_all = field_value(rd.fields[K("block_count")])[0]
    n_next = field_value(rd.fields[K("nextn_predict_layers")])[0] if K("nextn_predict_layers") in rd.fields else 0
    n_src = n_all - n_next
    iv_src = field_value(rd.fields[K("full_attention_interval")])[0]
    iv = a.interval or (2 if a.layers < 4 else iv_src)
    nl = a.layers
    keep_mtp = n_next > 0 and not a.no_mtp
    src_attn = [l for l in range(n_src) if (l + 1) % iv_src == 0]
    src_gdn = [l for l in range(n_src) if (l + 1) % iv_src != 0]
    lmap, ia, ig = {}, 0, 0
    for l in range(nl):
        if (l + 1) % iv == 0:
            lmap[l] = src_attn[ia % len(src_attn)]
            ia += 1
        else:
            lmap[l] = src_gdn[ig % len(src_gdn)]
            ig += 1
    if keep_mtp:
        lmap[nl] = n_src  # the nextn layer
    name = f"{arch}-tiny-L{nl}{'' if keep_mtp else '-nomtp'}-{a.weights}{a.seed}.gguf"
    out = a.out or os.path.join(OUTDIR, name)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    w = GGUFWriter(out + ".tmp", arch=arch)
    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", "general.name", "general.basename",
            "general.size_label", K("block_count"), K("full_attention_interval"), K("nextn_predict_layers")}
    for key, f in rd.fields.items():
        if key in skip or key.startswith("quantize.imatrix"):
            continue
        v, t, st = field_value(f)
        w.add_key_value(key, v, t, sub_type=st)
    w.add_name(f"WHIRL tiny {arch} {nl}L random ({a.weights} seed {a.seed})")
    w.add_uint32(K("block_count"), nl + (n_next if keep_mtp else 0))
    w.add_uint32(K("full_attention_interval"), iv)
    w.add_uint32(K("nextn_predict_layers"), n_next if keep_mtp else 0)
    w.add_string("whirl.tiny.source", os.path.basename(src))
    w.add_string("whirl.tiny.layer_map", ",".join(f"{k}<-{v}" for k, v in sorted(lmap.items())))

    by_name = {t.name: t for t in rd.tensors}
    tensors = []
    for t in rd.tensors:
        m = re.match(r"blk\.(\d+)\.(.+)", t.name)
        if not m:
            tensors.append((t.name, t))
    inv = {}
    for dst, s in sorted(lmap.items()):
        for t in rd.tensors:
            m = re.match(r"blk\.(\d+)\.(.+)", t.name)
            if m and int(m.group(1)) == s:
                tensors.append((f"blk.{dst}.{m.group(2)}", t))
    nbytes = 0
    for nm, t in tensors:
        arr, raw = randomize(t, a.seed, a.weights)
        w.add_tensor(nm, arr, raw_dtype=raw)
        nbytes += arr.nbytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(out + ".tmp", out)
    print(f"{out}: {len(tensors)} tensors, {nbytes / 2**20:.0f} MiB, layers {nl} (+{n_next if keep_mtp else 0} MTP), "
          f"interval {iv}, layer map {lmap}, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
