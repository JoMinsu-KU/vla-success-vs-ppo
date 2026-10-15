"""P2: 공개 체크포인트 3종을 curl로 받는다 (huggingface_hub 다운로더가 이 PC에서 멈추는 문제 우회).

- 파일 목록·LFS SHA-256·리비전은 HF API에서 받는다.
- 파일마다 curl 하나 (이어받기 -C -, 느려지면 끊고 다시: 60초 동안 100 KB/s 미만).
- 동시에 N개 파일. 끝나면 SHA-256(LFS) 또는 크기로 검증한다.
- 토큰은 ~/.cache/huggingface/token에서 읽어 헤더로만 넘긴다 (출력하지 않음).

사용: python p2_download_curl.py <저장 루트> <기록 JSON> [동시 수]
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi

REPOS = [
    "gen-robot/openvla-7b-rlvla-rl",       # P2에서 가장 먼저 쓸 체크포인트를 먼저 받는다
    "gen-robot/openvla-7b-rlvla-sft_16k",
    "gen-robot/openvla-7b-rlvla-warmup",
]
root = Path(sys.argv[1])
record_path = Path(sys.argv[2])
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
token_file = Path(os.environ.get("HF_TOKEN_PATH", Path.home() / ".cache" / "huggingface" / "token"))
token = token_file.read_text().strip() if token_file.exists() else None


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(job):
    repo, rev, name, size, sha = job
    dst = root / repo.split("/")[1] / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == size:
        return job, "exists"
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{name}"
    cmd = ["curl.exe", "-sS", "-L", "-C", "-", "--retry", "10", "--retry-delay", "5",
           "-Y", "102400", "-y", "60", "-o", str(dst), url]
    if token:
        cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
    for attempt in range(1, 201):
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cur = dst.stat().st_size if dst.exists() else 0
        if cur == size:
            return job, f"downloaded ({attempt} attempt(s))"
        if cur > size:  # 잘못 이어 붙은 경우 처음부터
            dst.unlink()
        log(f"  retry {repo}/{name}: {cur / 1e9:.2f}/{size / 1e9:.2f} GB")
        time.sleep(3)
    return job, "FAILED"


api = HfApi()
jobs, record = [], {}
for repo in REPOS:
    info = api.model_info(repo, files_metadata=True)
    files = {}
    for s in info.siblings:
        files[s.rfilename] = {"size": s.size, "lfs_sha256": s.lfs.sha256 if s.lfs else None}
        jobs.append((repo, info.sha, s.rfilename, s.size, s.lfs.sha256 if s.lfs else None))
    record[repo] = {"revision": info.sha, "local_path": str(root / repo.split("/")[1]), "files": files,
                    "total_gb": round(sum(v["size"] or 0 for v in files.values()) / 1e9, 2)}
    log(f"{repo} @ {info.sha}: {len(files)} files, {record[repo]['total_gb']} GB")

t0 = time.time()
with ThreadPoolExecutor(max_workers=workers) as ex:
    for job, status in ex.map(fetch, jobs):
        log(f"{status:>22}  {job[0]}/{job[2]}")

log("verifying SHA-256 ...")
ok = True
for repo, rev, name, size, sha in jobs:
    dst = root / repo.split("/")[1] / name
    good = dst.exists() and dst.stat().st_size == size and (sha is None or sha256_of(dst) == sha)
    record[repo]["files"][name]["verified"] = bool(good)
    ok &= good
    if not good:
        log(f"  VERIFY FAIL {repo}/{name}")
record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
log(f"ALL DONE ok={ok} in {(time.time() - t0) / 3600:.2f} h")
