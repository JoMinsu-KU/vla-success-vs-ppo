"""P2: 공개 체크포인트 3종을 HF 캐시(D:)로 내려받고 파일 목록·해시 정보를 기록한다."""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPOS = [
    "gen-robot/openvla-7b-rlvla-warmup",
    "gen-robot/openvla-7b-rlvla-sft_16k",
    "gen-robot/openvla-7b-rlvla-rl",
]
out = Path(sys.argv[1])
api = HfApi()
record = {}
for repo in REPOS:
    info = api.model_info(repo, files_metadata=True)
    print(f"== {repo} @ {info.sha}", flush=True)
    path = snapshot_download(repo, revision=info.sha)
    files = {s.rfilename: {"size": s.size, "lfs_sha256": (s.lfs.sha256 if s.lfs else None)} for s in info.siblings}
    record[repo] = {"revision": info.sha, "local_path": path, "files": files,
                    "total_gb": round(sum(v["size"] or 0 for v in files.values()) / 1e9, 2)}
    print(f"   done -> {path} ({record[repo]['total_gb']} GB)", flush=True)
out.write_text(json.dumps(record, indent=2), encoding="utf-8")
print("ALL DONE", flush=True)
