"""본실험(E1 1단계) 공통: 경로, G-PRE 설정, 맹검 입출력, n 표, 검정력.

맹검 원칙 (명세 4.2절, D-G2-4 조치 3):
  - 성공 여부·성공률·성공 수에 따라 움직이는 값(손실, KL, optimizer 스텝 수, 라운드·업데이트 시간 포함)은
    <작업 폴더>/blinded/ 아래에만 쓴다. 분석 담당자는 이 폴더를 열지 않는다.
  - 공개 출력(<작업 폴더>/public/)에는 실행 ID, 진행 라운드 수, 실행 전체 GPU시간 합계, 상태만 쓴다.
  - 실행 ID와 갈래의 대응은 <작업 폴더>/sealed/plan.json에만 있다 (해시는 public/plan_sha256.txt).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parent
for p in (REPO / "pilot" / "g2", REPO / "pilot" / "learner"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CFG = json.loads((STUDY / "gpre_config.json").read_text(encoding="utf-8"))
EXEC_CONDS = [c[0] for c in CFG["eval_conditions"] if c[3] == "execution"]


def work_dir() -> Path:
    return Path(os.environ.get("VLA_MAIN_WORK", str(Path.home() / "vla_main"))).expanduser()


def w1_ckpt() -> str:
    return os.environ.get("VLA_W1_CKPT", str(Path.home() / "vla_g2" / "checkpoints" / "openvla-7b-rlvla-warmup"))


def ensure_dirs(w: Path):
    for d in ("public", "blinded", "sealed", "runs", "logs"):
        (w / d).mkdir(parents=True, exist_ok=True)


def read_json(p: Path, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    os.replace(tmp, p)


def append_jsonl(p: Path, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(p: Path) -> list:
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def eval_rows(p: Path) -> list:
    """평가 기록 읽기. 배치 기록 중 끊겨 같은 에피소드가 두 번 있으면 마지막 것만 쓴다."""
    by = {}
    for r in read_jsonl(p):
        by[(r["cond"], r["k"])] = r
    return list(by.values())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def public_log(w: Path, msg: str):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with (w / "public" / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- 통계 (GPRE_simulation_v3과 같은 계산)
def required_n_paired(sd_d: float, delta: float, alpha: float, power: float = 0.8, n_max: int = 200) -> int:
    from scipy import stats
    for n in range(2, n_max + 1):
        df = n - 1
        ncp = delta / (sd_d / math.sqrt(n))
        if 1 - stats.nct.cdf(stats.t.ppf(1 - alpha, df), df, ncp) >= power:
            return n
    return n_max


def n_from_blinded_sd(sd_d: float) -> int:
    """맹검 SD_d를 0.1 단위로 반올림해 사전 고정 n 표(α 0.04, 검정력 0.8, 8~12)에서 n을 정한다."""
    s = max(round(sd_d, 1), 0.05)
    n = required_n_paired(s, CFG["delta_pp"], CFG["alpha_one_sided"])
    return int(min(CFG["n_max"], max(CFG["n_min"], n)))


def paired_power(sd_d: float, n: int, delta: float | None = None, alpha: float | None = None) -> float:
    """Δ = 0에서 비열등성 짝지은 t의 검정력 (고정 n 근사, 적응형 팽창 미반영)."""
    from scipy import stats
    delta = CFG["delta_pp"] if delta is None else delta
    alpha = CFG["alpha_one_sided"] if alpha is None else alpha
    df = n - 1
    ncp = delta / (max(sd_d, 1e-9) / math.sqrt(n))
    return float(1 - stats.nct.cdf(stats.t.ppf(1 - alpha, df), df, ncp))


def blinded_sd_d(block_pairs: list) -> float:
    """블록(학습 시드)마다 두 실행의 주지표 값 쌍 → √(Σd²/블록 수). 쌍 안의 순서(갈래)와 무관하다."""
    d2 = [(a - b) ** 2 for a, b in block_pairs]
    return math.sqrt(sum(d2) / len(d2))
