"""속도 개선(0-B LoRA 병합)의 결과 동일성 시험.

같은 프로세스에서:
  1. 원 경로(PEFT 래퍼 + 0-B LoRA)로 폐루프 에피소드 2개를 돌리며 관측·지시문·행동 토큰·log-prob·value를 기록
  2. LoRA 병합 후 같은 RNG 시드로 폐루프 에피소드 2개를 다시 돌려, 매 스텝의 관측·행동 토큰·log-prob·value 비교
     (계산이 같으면 RNG 소비 순서도 같으므로 80스텝 전체가 일치해야 한다)
  3. 스텝당 추론 시간 비교
통과 기준: 두 에피소드 160스텝 전체에서 관측과 행동 토큰이 완전히 일치 (log-prob·value 차이도 기록).
사용: python speed_equivalence_test.py <ckpt_dir> <out_json>
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_eval  # noqa: E402
import torch  # noqa: E402

ckpt, out_json = sys.argv[1], Path(sys.argv[2])
dev = torch.device("cuda:0")
policy, unnorm_state = p2_eval.load_policy(ckpt, "bridge_orig", 0.6, merge_lora=False)
TASK = ("PutOnPlateInScene25Carrot-v1", "train")


def rollout(policy, k):
    """seed 0, 에피소드 k를 폐루프로 돌려 (관측, 지시문, 토큰, logprob, value) 열을 돌려준다."""
    w = p2_eval.WinSimplerWrapper(TASK[0], unnorm_state, seed=0)
    torch.manual_seed(1234 + k)
    obs, instr, _ = w.reset_with_episode(TASK[1], p2_eval.episode_id_for(0, k))
    rec = []
    succ = None
    for t in range(p2_eval.EPISODE_LEN):
        with torch.no_grad():
            v, a, lp = policy.get_action(dict(image=obs.to(dev), task_description=instr), deterministic=True)
        rec.append((obs.clone(), instr, a.cpu().clone(), lp.float().cpu().clone(), v.float().cpu().clone()))
        obs, _r, _tr, info = w.step(a.cpu())
        if "episode" in info:
            succ = bool(info["episode"]["success"][0])
    w.env.close()
    return rec, succ


def step_time(policy, obs, instr, n=30):
    x = dict(image=obs.to(dev), task_description=instr)
    with torch.no_grad():
        for _ in range(3):
            policy.get_action(x, deterministic=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            policy.get_action(x, deterministic=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


# 1. 원 경로
base_runs = [rollout(policy, k) for k in (0, 1)]
obs0, instr0 = base_runs[0][0][10][0], base_runs[0][0][10][1]
t_base = step_time(policy, obs0, instr0)
print(f"[base] episodes success={[s for _, s in base_runs]}  step {t_base:.1f} ms", flush=True)

# 2. 병합
n_merged = p2_eval.merge_zero_lora(policy)
policy.prep_rollout()
t_merged = step_time(policy, obs0, instr0)
print(f"[merged] {n_merged} LoRA modules merged  step {t_merged:.1f} ms", flush=True)

# 2b. 폐루프 재실행 (같은 시드) → 원 경로와 행동 열 비교
merged_runs = [rollout(policy, k) for k in (0, 1)]
closed = []
for k in (0, 1):
    ra, sa = base_runs[k]
    rb, sb = merged_runs[k]
    tok_eq = all(torch.equal(x[2], y[2]) for x, y in zip(ra, rb))
    lp_d = max(float((x[3] - y[3]).abs().max()) for x, y in zip(ra, rb))
    v_d = max(float((x[4] - y[4]).abs().max()) for x, y in zip(ra, rb))
    obs_eq = all(torch.equal(x[0], y[0]) for x, y in zip(ra, rb))
    closed.append(dict(episode=k, steps=len(ra), action_tokens_identical=tok_eq, observations_identical=obs_eq,
                       logprob_max_abs_diff=lp_d, value_max_abs_diff=v_d, success_base=sa, success_merged=sb))
    print(f"[closed-loop ep{k}] tokens identical={tok_eq} obs identical={obs_eq} "
          f"logprob maxdiff={lp_d:.3g} value maxdiff={v_d:.3g} success {sa}->{sb}", flush=True)

result = dict(ckpt=ckpt, lora_modules_merged=n_merged, step_ms_base=round(t_base, 1),
              step_ms_merged=round(t_merged, 1), speedup=round(t_base / t_merged, 2), closed_loop=closed,
              passed=all(c["action_tokens_identical"] and c["observations_identical"] for c in closed))
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("RESULT", json.dumps(result), flush=True)
