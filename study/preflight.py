"""실행 전 점검: GPU, 디스크, 체크포인트, 패키지, 저장소 상태(태그 gpre). 실패 항목이 있으면 종료 코드 1."""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NEED_DISK_GB = 200
fails = []


def check(name, ok, detail=""):
    print(f"[{'OK' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


def main(smoke: bool):
    import torch
    check("cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        check("gpu", "H200" in name and mem > 120, f"{name}, {mem:.0f} GiB, 보이는 GPU {n}개")
        if n > 1:
            print("  참고: GPU가 여러 개 보인다. 실행기는 0번만 쓴다 (CUDA_VISIBLE_DEVICES로 하나만 보이게 권장)")
    W = C.work_dir()
    W.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(W).free / 2**30
    check("disk", free >= NEED_DISK_GB, f"{W}: 여유 {free:.0f} GiB (필요 {NEED_DISK_GB})")
    w1 = Path(C.w1_ckpt())
    check("W1 checkpoint", (w1 / "config.json").exists(), str(w1))
    for m in ("mani_skill", "simpler_env", "prismatic", "flash_attn", "transformers", "peft", "scipy"):
        try:
            importlib.import_module(m)
            check(f"import {m}", True)
        except Exception as e:  # noqa: BLE001
            check(f"import {m}", False, repr(e)[:120])
    check("cwd = RL4VLA/SimplerEnv", Path.cwd().name == "SimplerEnv", str(Path.cwd()))
    r = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "-q", "--verify", "refs/tags/gpre"], capture_output=True)
    if smoke:
        print(f"[INFO] 태그 gpre {'있음' if r.returncode == 0 else '없음'} (스모크 시험은 태그 없이 가능)")
    else:
        check("tag gpre", r.returncode == 0, "(G-PRE 고정 뒤에만 본 실행)")
    head = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(C.REPO), "status", "--porcelain", "--", "study", "pilot/g2", "pilot/learner"],
                           capture_output=True, text=True).stdout.strip()
    check("repo clean (study, pilot/g2, pilot/learner)", not dirty, f"HEAD {head}")
    print("PREFLIGHT", "PASS" if not fails else f"FAIL: {', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main("--smoke" in sys.argv)
