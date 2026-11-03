"""G2 S4: 공개 체크포인트를 통제 평가 경로로 평가 (명세 3.8절, C4).

- 조건마다 고정 episode_id 192개 (조건 이름·에피소드 번호에서 SHA-256으로 유도), 모든 체크포인트에 같은 집합
- 환경 64개 배치마다 장면 재구성(reconfigure) → 모든 에피소드가 새로 만든 장면에서 시작
- K0(top_k 0, top_p 1), 온도 0.6, 성공 = Y(마지막 스텝 success ∧ 집기 래치). 원 기준(success)과 any-step도 기록
- attention: 원 코드와 같은 flash_attention_2 (실제 구현 기록)
- unnorm key: 공개 체크포인트가 학습된 통계를 쓴다 (RL·웜업 bridge_orig, SFT sft). C1 통일은 자체 학습분에만 해당
사용: python g2_s4_eval_controlled.py <체크포인트 폴더> <out_dir> [--conds core|all]
결과: <out_dir>/g2_s4_episodes.jsonl (에피소드별), g2_s4_summary.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_gpu as G  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CORE = [("ID (Carrot train)", "PutOnPlateInScene25Carrot-v1", "train"),
        ("Position", "PutOnPlateInScene25Position-v1", "test"),
        ("EEPose", "PutOnPlateInScene25EEPose-v1", "test"),
        ("PositionChangeTo", "PutOnPlateInScene25PositionChangeTo-v1", "test"),
        ("New object (Carrot test)", "PutOnPlateInScene25Carrot-v1", "test")]
CKPTS = [("rl", "openvla-7b-rlvla-rl", "bridge_orig"), ("sft", "openvla-7b-rlvla-sft_16k", "sft"),
         ("warmup", "openvla-7b-rlvla-warmup", "bridge_orig")]
N_EP = 192


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ckpts", default="rl,sft,warmup")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    epf = out / "g2_s4_episodes.jsonl"
    done = set()
    if epf.exists():
        for line in epf.read_text().splitlines():
            d = json.loads(line)
            done.add((d["ckpt"], d["cond"], d["batch"]))
    # 고정 집합 (해시로 기록)
    sets = {lab: [G.episode_id_for(f"g2-eval-{env}-{split}", k) for k in range(N_EP)] for lab, env, split in CORE}
    import hashlib
    (out / "g2_s4_episode_sets.json").write_text(json.dumps(dict(
        sets=sets, sha256={k: hashlib.sha256(json.dumps(v).encode()).hexdigest() for k, v in sets.items()}), indent=1))
    timing = []
    for name, sub, unnorm in [c for c in CKPTS if c[0] in a.ckpts.split(",")]:
        todo = [(lab, env, split, b) for lab, env, split in CORE for b in range(3) if (name, lab, b) not in done]
        if not todo:
            continue
        t0 = time.time()
        policy = G.build_policy(str(Path(a.ckpt_dir) / sub), unnorm, attn="flash_attention_2", seed=0)
        G.set_top_k(policy, "K0")
        ustate = policy.vla.get_action_stats(unnorm)
        print(f"[load] {name} {time.time() - t0:.0f}s attn={G.attn_impls(policy)}", flush=True)
        env, cur = None, None
        for lab, env_id, split, b in todo:
            if env_id != cur:
                if env is not None:
                    env.close()
                env, cur = G.GpuEnv(env_id, ustate, 64, seed=0), env_id
            eids = sets[lab][64 * b:64 * (b + 1)]
            tm = {}
            r = G.rollout(policy, env, split, eids, deterministic=True, rng_seed=1000 + b, timing=tm)
            sel = r["selected"]
            with epf.open("a") as f:
                for i in range(64):
                    f.write(json.dumps(dict(ckpt=name, cond=lab, env_id=env_id, obj_set=split, batch=b, k=64 * b + i,
                                            episode_id=int(eids[i]), instruction=r["instr"][i],
                                            Y=bool(r["Y"][i]), success=bool(r["success"][-1, i]),
                                            success_any=bool(r["success"][:, i].any()),
                                            grasp_latch=bool(r["grasp_latch"][-1, i]),
                                            **{k: int(v[i]) for k, v in sel.items()})) + "\n")
            timing.append(dict(ckpt=name, cond=lab, batch=b, **{k: round(v, 1) for k, v in tm.items()}))
            print(f"[{name} {lab} b{b}] Y {r['Y'].sum()}/64 success {r['success'][-1].sum()}/64 "
                  f"{tm['total_s']:.0f}s (policy {tm['policy_s']:.0f}, sim {tm['sim_s']:.0f})", flush=True)
        env.close()
        del policy
        torch.cuda.empty_cache()
    with (out / "g2_s4_timing.jsonl").open("a") as f:
        for t in timing:
            f.write(json.dumps(t) + "\n")
    summarize(out)


def summarize(out: Path):
    rows = [json.loads(x) for x in (out / "g2_s4_episodes.jsonl").read_text().splitlines()]
    S = {}
    for lab, _e, _s in CORE:
        S[lab] = {}
        by = {}
        for ck in ("rl", "sft", "warmup"):
            rr = sorted([r for r in rows if r["ckpt"] == ck and r["cond"] == lab], key=lambda r: r["k"])
            if not rr:
                continue
            by[ck] = rr
            n = len(rr)
            for key in ("Y", "success", "success_any"):
                p = sum(r[key] for r in rr) / n
                S[lab].setdefault(ck, {})[key] = round(100 * p, 1)
            p = S[lab][ck]["Y"] / 100
            S[lab][ck].update(n=n, se_Y=round(100 * math.sqrt(p * (1 - p) / n), 1),
                              per_batch_Y=[round(100 * np.mean([r["Y"] for r in rr if r["batch"] == b]), 1) for b in range(3)])
        if "rl" in by and "sft" in by and len(by["rl"]) == len(by["sft"]):
            assert [r["episode_id"] for r in by["rl"]] == [r["episode_id"] for r in by["sft"]]
            b = sum(x["Y"] and not y["Y"] for x, y in zip(by["rl"], by["sft"]))
            c = sum(y["Y"] and not x["Y"] for x, y in zip(by["rl"], by["sft"]))
            S[lab]["rl_minus_sft_Y"] = round(S[lab]["rl"]["Y"] - S[lab]["sft"]["Y"], 1)
            S[lab]["paired"] = dict(rl_only=b, sft_only=c, mcnemar_p=mcnemar(b, c))
    (out / "g2_s4_summary.json").write_text(json.dumps(S, indent=2, ensure_ascii=False))
    for lab, v in S.items():
        print(lab, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()
