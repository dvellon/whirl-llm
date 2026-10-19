"""REQUANT-1: check every downloaded file against the Hugging Face API listing (size; SHA256 for LFS files).
Usage: python -I verify_src.py API.json SRC_DIR OUT.txt   (API.json = /api/models/<repo>/revision/<sha>?blobs=true)"""
import hashlib, json, os, sys
api, src, out = sys.argv[1:4]
d = json.load(open(api, encoding="utf-8"))
rows, bad, n = [], 0, 0
for f in d["siblings"]:
    p = os.path.join(src, f["rfilename"])
    if not os.path.exists(p):
        rows.append(f"SKIP {f['rfilename']} (not downloaded)"); continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
    sha, size = h.hexdigest(), os.path.getsize(p)
    lfs = (f.get("lfs") or {}).get("sha256")
    ok = size == f.get("size") and (lfs is None or lfs == sha)
    bad += not ok; n += 1
    rows.append(f"{'OK ' if ok else 'BAD'} {sha}  {size}  {f['rfilename']}  {'lfs-match' if lfs else 'size-match (no LFS hash in API)'}")
rows.append(f"repo {d['id']} rev {d['sha']}: {n} files checked, {bad} bad")
open(out, "w", encoding="utf-8").write("\n".join(rows) + "\n")
print(rows[-1])
sys.exit(1 if bad else 0)
