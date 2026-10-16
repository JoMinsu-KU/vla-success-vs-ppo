"""배치 실행기 시험: (1) 배치 추론과 배치 1 추론의 수치 차이, (2) 환경 N개 병렬의 GPU 메모리·처리량.

(1) 실제 관측 N개에 대해 첫 행동 토큰의 256개 행동 로짓을 배치 1과 배치 N으로 계산해 비교한다
    (생성 전, 같은 입력). greedy 토큰 일치율은 거의 같은 확률의 bin 사이에서 작은 수치 차이로 argmax가 바뀌고
    자기회귀로 이후 토큰이 모두 달라지므로 지표로 부적절하다. 대신 첫 토큰 분포의 KL과 최대 로짓 차이를 잰다.
    대조군: 같은 배치 1 계산을 두 번 해서 생기는 차이 (같은 경로의 비결정성).
(2) 온도 0.6, K50으로 에피소드 2N개를 돌려 에피소드당 시간과 GPU 메모리를 잰다.
사용: python batch_test.py <ckpt_dir> <N> <out_json>
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_eval  # noqa: E402
import batched_eval as be  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def first_token_logits(policy, obs_list, instr_list, dev):
    images = torch.from_numpy(np.stack(obs_list)).to(dev)
    feats = policy._preprocess_obs(dict(image=images, task_description=list(instr_list)))
    with torch.no_grad():
        out = policy.vla(input_ids=feats["input_ids"], attention_mask=feats["attention_mask"],
                         pixel_values=feats["pixel_values"])
    return out.logits[:, -1, 32000 - 256:32000].float()  # [B, 256]


def compare(a, b):
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (la.exp() * (la - lb)).sum(-1)
    return dict(max_abs_logit_diff=float((a - b).abs().max()), mean_kl=float(kl.mean()), max_kl=float(kl.max()),
                argmax_agree=float((a.argmax(-1) == b.argmax(-1)).float().mean()))


def gpu_mem_mib():
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(q.stdout.strip().splitlines()[0])


if __name__ == "__main__":
    ckpt, N, out_json = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    policy, unnorm_state = p2_eval.load_policy(ckpt, "bridge_orig", 0.6, merge_lora=True)
    be.set_top_k(policy, "K50")
    dev = torch.device("cuda:0")
    mem_model = gpu_mem_mib()
    pool = be.EnvPool(N, unnorm_state, scale_seed=0)
    specs = [dict(env_id="PutOnPlateInScene25Carrot-v1", obj_set="train", episode_id=p2_eval.episode_id_for(1, k))
             for k in range(N)]
    resets = pool.reset(specs)
    mem_envs = gpu_mem_mib()
    print(f"[mem] after model {mem_model} MiB, after {N} envs {mem_envs} MiB "
          f"(~{(mem_envs - mem_model) / N:.0f} MiB per env)", flush=True)
    obs = [r[0] for r in resets]
    instr = [r[1] for r in resets]

    single = torch.cat([first_token_logits(policy, [o], [s], dev) for o, s in zip(obs, instr)])
    single2 = torch.cat([first_token_logits(policy, [o], [s], dev) for o, s in zip(obs, instr)])
    batched = first_token_logits(policy, obs, instr, dev)
    c_batch = compare(single, batched)
    c_repeat = compare(single, single2)
    print(f"[batch vs single] {c_batch}", flush=True)
    print(f"[single vs single repeat] {c_repeat}", flush=True)

    recs = []
    t0 = time.time()
    run_specs_list = [dict(env_id="PutOnPlateInScene25Carrot-v1", obj_set="train", seed=1, episode=k,
                           episode_id=p2_eval.episode_id_for(1, k)) for k in range(2 * N)]
    be.run_specs(policy, pool, run_specs_list, "bridge_orig", 0.6, rng_seed=777,
                 on_episode=lambda s, r: recs.append(r), log=lambda m: print(m, flush=True))
    dt = time.time() - t0
    mem_peak = gpu_mem_mib()
    pool.close()
    res = dict(ckpt=ckpt, N=N, mem_after_model_mib=mem_model, mem_after_envs_mib=mem_envs, mem_end_mib=mem_peak,
               batch_vs_single=c_batch, single_repeat=c_repeat, episodes=len(recs),
               sec_per_episode=round(dt / len(recs), 2), success=sum(r["success"] for r in recs))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("RESULT", json.dumps(res), flush=True)
