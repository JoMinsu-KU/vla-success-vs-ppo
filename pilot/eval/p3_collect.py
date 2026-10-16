"""P3 웜업 공급 측정 수집 (연구 명세 v0.9, 6.3절 P3, 3.7절 전환 규칙, C4 top-k 선택 규칙).

- 정책: 공개 웜업 W1 (gen-robot/openvla-7b-rlvla-warmup, bridge_orig), 0-B LoRA 병합(결과 동일 확인됨).
- 환경: 학습 과제 분포 PutOnPlateInScene25Main-v3, obj_set=train (물체 16 × 테이블 16).
- 초기 상태: 물체 16 × 테이블 16 × 4회 완전 교차 = 1,024 에피소드. 위치·자세는 고정 시드에서 무작위.
  episode_id = 물체·(lp·lo·l1·l2) + 테이블·(l1·l2) + 위치·l2 + 자세  (lp = 1, 접시 하나)
- 설정 3개, 같은 1,024 초기 상태에서 (짝지은 비교):
    T1_K0  : 온도 1.0, top_k 0   (v0.9 기본안 K0, 수집 설정)
    T1_K50 : 온도 1.0, top_k 50  (원 코드 수집 설정 = 대안 K50의 샘플링)
    T06_K50: 온도 0.6, top_k 50  (원 평가 설정, 평가 성공률로 공급량을 추정할 때의 오차 확인)
- 결과를 보고 멈추는 규칙 없음 (고정 에피소드 수).

사용: python p3_collect.py <ckpt_dir> <out_jsonl> [환경 수 N] [설정들, 쉼표]
같은 출력 파일로 다시 실행하면 끝난 (설정, 초기 상태)는 건너뛴다.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_eval  # noqa: E402
import batched_eval as be  # noqa: E402
import numpy as np  # noqa: E402

ENV_ID = "PutOnPlateInScene25Main-v3"
SETTINGS = {"T1_K0": (1.0, "K0"), "T1_K50": (1.0, "K50"), "T06_K50": (0.6, "K50")}
N_OBJ, N_TABLE, N_REP = 16, 16, 4

if __name__ == "__main__":
    ckpt, out = sys.argv[1], Path(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    settings = sys.argv[4].split(",") if len(sys.argv) > 4 else list(SETTINGS)
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, unnorm_state = p2_eval.load_policy(ckpt, "bridge_orig", 0.6, merge_lora=True)
    pool = be.EnvPool(N, unnorm_state, scale_seed=0)
    d = pool.dims(ENV_ID)
    l1, l2, lp = d["n_pos"], d["n_quat"], d["n_plate"]
    assert d["n_carrot"] >= N_OBJ and lp == 1, d
    print(f"[dims] positions {l1}, quats {l2}, objects {d['n_carrot']}, plates {lp}", flush=True)

    rng = np.random.default_rng(3)  # 위치·자세 선택용 (설정과 무관하게 고정)
    base_specs = []
    for rep in range(N_REP):
        for obj in range(N_OBJ):
            for table in range(N_TABLE):
                pos, quat = int(rng.integers(l1)), int(rng.integers(l2))
                eid = obj * (lp * N_TABLE * l1 * l2) + table * (l1 * l2) + pos * l2 + quat
                base_specs.append(dict(env_id=ENV_ID, obj_set="train", episode_id=eid, p3_index=len(base_specs),
                                       rep=rep, obj=obj, table=table, pos=pos, quat=quat,
                                       obj_name=d["carrot_names"][obj]))
    # 한 웨이브 안에 같은 물체·테이블이 몰리지 않게 순서를 섞되, 시드 고정
    order = np.random.default_rng(4).permutation(len(base_specs))
    base_specs = [base_specs[i] for i in order]

    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["setting"], r["p3_index"]))

    for si, name in enumerate(settings):
        temp, k = SETTINGS[name]
        be.set_top_k(policy, k)
        specs = [dict(s, setting=name, top_k_mode=k) for s in base_specs if (name, s["p3_index"]) not in done]
        print(f"[setting {name}] temperature {temp}, {k}: {len(specs)} episodes to run", flush=True)
        if not specs:
            continue
        t0 = time.time()

        def save(_s, rec):
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        be.run_specs(policy, pool, specs, "bridge_orig", temp, rng_seed=1_000_000 * (si + 1), on_episode=save,
                     log=lambda m: print(f"[{name}] {m}", flush=True))
        print(f"[setting {name}] done in {(time.time() - t0) / 3600:.2f} h", flush=True)
    pool.close()
    print("ALL DONE", flush=True)
