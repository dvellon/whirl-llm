"""REQUANT-1: Ornith imatrix calibration text = bartowski calibration-v6 (English prose + chat/tool-call
conversations) + klgate code.txt[70000:] + zh.txt[40000:], with every line that also occurs in the KL corpus
(kl_mix.txt) removed, then checked: no 120-char window of kl_mix (every 40th position) occurs in the result.
Usage: python -I make_orn_calib.py BARTOWSKI.txt KL_MIX.txt OUT.txt"""
import sys
bwp, klp, outp = sys.argv[1:4]
src = r"C:\ClaudeWorkspace\whirl-work\klgate\corpus"
code = open(src + r"\code.txt", encoding="utf-8").read()[70000:]
zh = open(src + r"\zh.txt", encoding="utf-8").read()[40000:]
bw = open(bwp, encoding="utf-8").read()
kl = open(klp, encoding="utf-8").read()
kl_lines = {l.strip() for l in kl.splitlines() if len(l.strip()) >= 8}
def clean(t):
    keep, drop = [], 0
    for l in t.splitlines():
        if l.strip() in kl_lines: drop += 1
        else: keep.append(l)
    return "\n".join(keep), drop
parts = []
for name, t in (("bartowski", bw), ("code", code), ("zh", zh)):
    c, d = clean(t); parts.append(c); print(f"{name}: {len(t)} chars, dropped {d} lines shared with kl_mix, kept {len(c)}")
out = "\n\n".join(parts)
hits = [i for i in range(0, len(kl) - 120, 40) if kl[i:i + 120] in out]
print(f"calibration {len(out)} chars; kl_mix 120-char windows found in it: {len(hits)} of {len(range(0, len(kl) - 120, 40))}")
open(outp, "w", encoding="utf-8", newline="\n").write(out)
sys.exit(1 if hits else 0)
