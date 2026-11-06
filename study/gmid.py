"""G-mid 맹검 분산 재추정 (G-PRE 초안 1절, pilot/stats/GPRE_simulation_v3.md).

블록 = 학습 시드. 블록마다 두 E1 실행의 주지표(최종 체크포인트, 실행 3조건 평균 성공률 %)를 읽어
SD_d = √(Σ d_s² / 블록 수)를 계산한다. d²는 두 실행의 순서(갈래)와 무관하므로 갈래 대응표를 읽지 않는다.
공개 출력: SD_d(0.1 단위), 사전 고정 n 표의 n, 그 n에서의 예상 검정력(Δ = 0, α 0.04, 고정 n 근사), 사전 규칙 판정.
갈래별 평균·부호·블록별 값은 출력하지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def primary_endpoint(rows: list) -> float:
    """실행 3조건 각 성공률(%)의 평균."""
    per = []
    for c in C.EXEC_CONDS:
        ys = [r["Y"] for r in rows if r["cond"] == c]
        if not ys:
            raise ValueError(f"missing condition {c}")
        per.append(100 * sum(ys) / len(ys))
    return sum(per) / len(per)


def main(work: Path, index: dict, seeds: list, eval_name: str = "final") -> dict:
    pairs = []
    for s in seeds:
        rids = [rid for rid, v in index.items() if v["kind"] == "e1_train" and v["seed"] == s]
        if len(rids) != 2:
            raise SystemExit(f"seed {s}: E1 실행이 2개가 아니다 ({len(rids)})")
        vals = [primary_endpoint(C.eval_rows(work / "blinded" / rid / f"eval_e1_{eval_name}.jsonl")) for rid in rids]
        pairs.append((vals[0], vals[1]))
    sd = C.blinded_sd_d(pairs)
    sd_r = round(sd, 1)
    n = C.n_from_blinded_sd(sd_r)
    pw = C.paired_power(sd_r, n)
    pw12 = C.paired_power(sd_r, C.CFG["n_max"])
    thr = C.CFG["gmid_power_threshold"]
    out = dict(blocks=len(pairs), sd_d=sd_r, n=n, power_at_n=round(pw, 3), power_at_12=round(pw12, 3),
               threshold=thr, verdict="plan" if pw >= thr else "reduce",
               note="power: Δ=0, 단측 α 0.04, 고정 n 근사 (적응형 팽창 미반영). verdict 'reduce'는 G-PRE 초안 7절의 축소 규칙 적용",
               decided=C.now())
    C.write_json(work / "public" / "gmid.json", out)
    return out
