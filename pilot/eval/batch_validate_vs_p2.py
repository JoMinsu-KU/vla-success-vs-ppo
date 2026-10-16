"""배치 실행기 검증: P2와 같은 초기 상태(seed 0, 에피소드 0~63)를 배치 실행기로 다시 돌려 P2 결과와 짝지어 비교.

P2: 원 경로(PEFT 래퍼, 배치 1, 환경 1개). 여기: LoRA 병합(비트 동일 확인) + 배치 N + 환경 프로세스 N개.
배치 수치 차이(첫 토큰 KL 약 0.002)와 샘플링 RNG 소비 순서가 달라 에피소드 단위로 같을 필요는 없고,
성공률이 통계적으로 같은지(정확 McNemar)를 본다.
사용: python batch_validate_vs_p2.py <ckpt_dir> <p2_results.jsonl> <N> <out_jsonl> [task env_id:split]
"""
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_eval  # noqa: E402
import batched_eval as be  # noqa: E402

if __name__ == "__main__":
    ckpt, p2_path, N, out = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    env_id, split = (sys.argv[5].split(":") if len(sys.argv) > 5 else ("PutOnPlateInScene25Carrot-v1", "train"))
    name = Path(ckpt).name
    p2 = {}
    for line in p2_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if Path(r["ckpt"]).name == name and r["env_id"] == env_id and r["obj_set"] == split and r["seed"] == 0:
            p2[r["episode"]] = r["success"]
    assert len(p2) == 64, len(p2)

    policy, unnorm_state = p2_eval.load_policy(ckpt, "bridge_orig", 0.6, merge_lora=True)
    be.set_top_k(policy, "K50")
    pool = be.EnvPool(N, unnorm_state, scale_seed=0)
    specs = [dict(env_id=env_id, obj_set=split, seed=0, episode=k, episode_id=p2_eval.episode_id_for(0, k))
             for k in range(64)]
    recs = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        def save(_s, r):
            recs.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        be.run_specs(policy, pool, specs, "bridge_orig", 0.6, rng_seed=0, on_episode=save,
                     log=lambda m: print(m, flush=True))
    pool.close()

    b = {r["episode"]: r["success"] for r in recs}
    n10 = sum(1 for k in range(64) if p2[k] and not b[k])
    n01 = sum(1 for k in range(64) if b[k] and not p2[k])
    n = n10 + n01
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    res = dict(task=f"{env_id}/{split}", p2_success=sum(p2.values()), batched_success=sum(b.values()), n=64,
               p2_only=n10, batched_only=n01, mcnemar_p=p,
               sec_per_episode=round(sum(r["wave_sec"] for r in recs) / len(recs) / 1, 2))
    print("RESULT", json.dumps(res), flush=True)
    (out.with_suffix(".summary.json")).write_text(json.dumps(res, indent=2), encoding="utf-8")
