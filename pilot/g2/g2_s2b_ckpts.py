"""G2 S2b: 공개 체크포인트 3종을 4090 파일럿과 같은 revision으로 받고 크기·LFS SHA-256을 대조한다.
사용: python g2_s2b_ckpts.py <p2_checkpoints.json> <체크포인트 폴더> <결과 json>
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ref = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
out = Path(sys.argv[3])
res = {}
for repo, meta in ref.items():
    local = root / repo.split("/")[1]
    print(f"== {repo} @ {meta['revision']} -> {local}", flush=True)
    snapshot_download(repo, revision=meta["revision"], local_dir=str(local), max_workers=8)
    files = {}
    for name, f in meta["files"].items():
        p = local / name
        r = {"exists": p.exists()}
        if p.exists():
            r["size_ok"] = p.stat().st_size == f["size"]
            if f["lfs_sha256"]:
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 24), b""):
                        h.update(b)
                r["sha256_ok"] = h.hexdigest() == f["lfs_sha256"]
        files[name] = r
    ok = all(r["exists"] and r["size_ok"] and r.get("sha256_ok", True) for r in files.values())
    res[repo] = {"revision": meta["revision"], "local_path": str(local), "all_ok": ok, "files": files}
    print(f"   all_ok={ok}", flush=True)
out.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("S2b verdict:", {k: v["all_ok"] for k, v in res.items()}, flush=True)
