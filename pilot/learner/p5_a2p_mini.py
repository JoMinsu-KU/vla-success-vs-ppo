"""P5: A2⁺(성공 중심 온라인 학습) 미니 실행 (연구 명세 v0.9, 6.3절 P5, 3.3절).

- 출발: 병합 W1(공개 웜업) + 원 PPO 레시피의 새 LoRA(rank 32, alpha 16, gaussian A·0 B). 초기화 시드 = 학습 시드(C2 v0.9).
- 라운드: 현재 정책으로 64 에피소드 수집 (온도 1.0, top-k 방식은 인자. P3 규칙의 결과를 쓴다) →
  성공(Y = 마지막 스텝 success) 궤적의 전이만 모아 토큰 평균 NLL로 업데이트. 0-성공 라운드는 업데이트 없음.
- 업데이트 방식 U-prop: 성공 전이를 한 번 훑는다. optimizer 스텝마다 전이 160개(원 미니배치 8 × 누적 20과 같은 수),
  마지막 부분 묶음은 실제 개수로 나눈다(controlled). 메모리 때문에 미니배치를 mb로 나눠 누적한다(수학적으로 같은 평균).
- 기울기 클리핑 노름 10, AdamW(lr 1e-4, β 0.9/0.999 = 원 PPO 설정), KL 안정화 장치(라운드 시작 정책 대비 k3 KL,
  고정 부분집합 32 전이, KL_max 초과 시 그 라운드의 남은 스텝 생략)는 기록과 함께 작동.
- 성능 향상 여부는 판정 기준이 아니다 (6.1절). 통과 기준: NaN·OOM 없음, 기울기 유한·기록, 학습 데이터에서 손실 감소,
  저장 수가 수집 수와 일치.

사용: python p5_a2p_mini.py <W1 ckpt> <out_dir> --seeds 0 1 --rounds 10 --envs 3 --topk K0 --mb 4
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "learner"))
import p2_eval  # noqa: E402  (win_compat, sdpa 패치 포함)
import batched_eval as be  # noqa: E402
import core  # noqa: E402
import hashlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ENV_ID = "PutOnPlateInScene25Main-v3"


def eid(seed, rnd, k):
    h = hashlib.sha256(f"p5-seed{seed}-round{rnd}-ep{k}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 62) - 1)


def build_policy(ckpt, seed, lr):
    from simpler_env.policies.openvla import openvla_train as ot
    from prismatic.extern.hf import modeling_prismatic as mp  # flash-attn 대신 sdpa (p2_eval과 같은 처리)
    cls = mp.OpenVLAForActionPredictionWithValueHead
    if not getattr(cls, "_p2_attn_patched", False):
        orig = cls.from_pretrained.__func__

        @classmethod
        def fp(c, *a, **kw):
            if kw.get("attn_implementation") == "flash_attention_2":
                kw["attn_implementation"] = "sdpa"
            return orig(c, *a, **kw)
        cls.from_pretrained = fp
        cls._p2_attn_patched = True
    args = SimpleNamespace(vla_path=ckpt, vla_unnorm_key="bridge_orig", vla_load_path="", vla_lora_rank=32,
                           vla_lr=lr, vla_vhlr=3e-3, vla_optim_beta1=0.9, vla_optim_beta2=0.999,
                           vla_temperature=1.0, vla_temperature_eval=0.6)
    torch.manual_seed(seed)  # LoRA A 초기화 시드 = 학습 시드
    policy = ot.OpenVLAPolicy(args, device_id=0)
    lora_params = [p for n, p in policy.vla.named_parameters() if "lora_" in n and p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr, betas=(0.9, 0.999))
    return policy, lora_params, opt


def collect(policy, make_pool, seed, rnd, n_eps, topk, log, max_retries=3):
    """환경 풀을 만들어 64 에피소드를 수집한다. 작업자가 죽으면(렌더러 충돌 등) 풀을 다시 만들고 그 웨이브만 다시 한다.
    다시 한 웨이브는 같은 초기 상태·같은 RNG 시드로 처음부터 돌리므로 결과 분포는 같다. 재시도 수는 기록한다."""
    be.set_top_k(policy, topk)
    specs = [dict(env_id=ENV_ID, obj_set="train", episode_id=eid(seed, rnd, k), seed=seed, round=rnd, episode=k)
             for k in range(n_eps)]
    dev = torch.device("cuda:0")
    pool = make_pool()
    n = len(pool.conns)
    trajs, recs, retries = [], [], 0
    policy.prep_rollout()
    for w0 in range(0, n_eps, n):
        wave = specs[w0:w0 + n]
        for attempt in range(max_retries + 1):
            try:
                torch.manual_seed(10_000_000 * seed + 10_000 * rnd + w0)
                resets = pool.reset(wave)
                obs = [r[0] for r in resets]
                instr = [r[1] for r in resets]
                buf = [dict(obs=[], tok=[], lp=[]) for _ in wave]
                last = None
                for t in range(80):
                    toks, lps = be.policy_step(policy, obs, instr, "bridge_orig", 1.0, dev)
                    for i in range(len(wave)):
                        buf[i]["obs"].append(obs[i])
                        buf[i]["tok"].append(toks[i])
                        buf[i]["lp"].append(lps[i])
                    outs = pool.step(list(toks))
                    obs = [o[0] for o in outs]
                    last = [o[1] for o in outs]
                break
            except (EOFError, BrokenPipeError, ConnectionResetError, OSError) as e:
                retries += 1
                log(f"  [worker failure] wave {w0 // n} attempt {attempt + 1}: {type(e).__name__}; restarting env pool")
                try:
                    pool.close()
                except Exception:
                    pass
                for pr in pool.procs:
                    if pr.is_alive():
                        pr.kill()
                torch.cuda.empty_cache()
                time.sleep(5)
                pool = make_pool()
                if attempt == max_retries:
                    raise
        for i, s in enumerate(wave):
            Y = bool(last[i]["success"])
            recs.append(dict(s, instruction=instr[i], success=Y, grasped_latch=last[i]["grasped_latch"]))
            if Y:
                trajs.append(dict(obs=np.stack(buf[i]["obs"]), tok=np.stack(buf[i]["tok"]),
                                  lp=np.array(buf[i]["lp"], dtype=np.float32), instruction=instr[i]))
    pool.close()
    return trajs, recs, retries


def joint_logprob(policy, obs, instr, tok):
    dev = torch.device("cuda:0")
    x = dict(image=torch.from_numpy(obs).to(dev), task_description=list(instr))
    lp, _ent, _v = policy.evaluate_actions(x, torch.from_numpy(tok).to(dev))
    return lp  # [B, 1] joint log-prob (7 토큰 합)


def train_round(policy, lora_params, opt, trajs, mb, kl_max, rng, log):
    """U-prop: 성공 전이 전체를 무작위 순서로 한 번 훑는다. 160개마다 optimizer 스텝."""
    O = np.concatenate([t["obs"] for t in trajs])
    K = np.concatenate([t["tok"] for t in trajs])
    I = [t["instruction"] for t in trajs for _ in range(80)]
    N = len(O)
    order = rng.permutation(N)
    divisors = core.chunk_loss_divisors(N, minibatch=8, accum=20, controlled=True)  # 스텝마다 미니배치(8) 수
    # KL·손실 감시용 고정 부분집합
    sub = rng.choice(N, size=min(32, N), replace=False)

    def eval_subset():
        policy.prep_rollout()
        out = []
        with torch.no_grad():
            for j in range(0, len(sub), mb):
                idx = sub[j:j + mb]
                out.append(joint_logprob(policy, O[idx], [I[i] for i in idx], K[idx]).float().squeeze(-1))
        return torch.cat(out)

    lp_start = eval_subset()
    nll_start = float(-lp_start.mean() / 7)
    steps, stopped_by_kl = [], False
    pos = 0
    for s, n_mb8 in enumerate(divisors):
        n_tr = min(n_mb8 * 8, N - pos)
        idx_step = order[pos:pos + n_tr]
        pos += n_tr
        policy.prep_training()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for j in range(0, n_tr, mb):
            idx = idx_step[j:j + mb]
            lp = joint_logprob(policy, O[idx], [I[i] for i in idx], K[idx])
            loss = (-lp.float() / 7).sum() / n_tr  # 스텝의 전이 평균 토큰 NLL (부분 묶음은 실제 개수로 나눔)
            loss.backward()
            loss_sum += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(lora_params, 10.0))
        finite = math.isfinite(gn) and math.isfinite(loss_sum)
        if finite:
            opt.step()
        lp_now = eval_subset()
        kl = core.kl_estimate(lp_start, lp_now)
        steps.append(dict(step=s, transitions=n_tr, loss=loss_sum, grad_norm=gn, finite=finite, kl=kl))
        log(f"    step {s + 1}/{len(divisors)} n={n_tr} loss {loss_sum:.4f} grad_norm {gn:.3f} kl {kl:.4f}")
        if not finite:
            break
        if core.should_stop_round(kl, kl_max):
            stopped_by_kl = True
            break
    nll_end = float(-eval_subset().mean() / 7)
    return dict(n_transitions=N, n_steps_planned=len(divisors), steps=steps, stopped_by_kl=stopped_by_kl,
                subset_nll_before=nll_start, subset_nll_after=nll_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--envs", type=int, default=3)
    ap.add_argument("--topk", default="K0", choices=["K0", "K50"])
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-max", type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "p5_log.txt").open("a", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    for seed in a.seeds:
        policy, lora_params, opt = build_policy(a.ckpt, seed, a.lr)
        unnorm = policy.vla.get_action_stats("bridge_orig")
        log(f"[seed {seed}] LoRA params {sum(p.numel() for p in lora_params):,}, cuda {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
        rng = np.random.default_rng(seed)
        for rnd in range(a.rounds):
            t0 = time.time()
            torch.cuda.empty_cache()  # 학습 때 캐시된 GPU 메모리를 돌려줘야 환경 렌더 컨텍스트가 들어간다
            trajs, recs, retries = collect(policy, lambda: be.EnvPool(a.envs, unnorm, scale_seed=seed),
                                           seed, rnd, a.episodes, a.topk, log)
            t_col = time.time() - t0
            with (out / "p5_episodes.jsonl").open("a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_succ = len(trajs)
            assert n_succ == sum(r["success"] for r in recs)
            row = dict(seed=seed, round=rnd, episodes=len(recs), successes=n_succ, collect_sec=round(t_col, 1),
                       worker_retries=retries,
                       stored_bytes=int(sum(t["obs"].nbytes + t["tok"].nbytes for t in trajs)))
            log(f"[seed {seed} round {rnd}] successes {n_succ}/{len(recs)} collect {t_col / 60:.1f} min")
            if n_succ == 0:
                row.update(zero_success_round=True, update=None)
            else:
                torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                upd = train_round(policy, lora_params, opt, trajs, a.mb, a.kl_max, rng, log)
                row.update(zero_success_round=False, update=upd, train_sec=round(time.time() - t1, 1),
                           peak_train_cuda_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
                torch.cuda.empty_cache()
                log(f"[seed {seed} round {rnd}] steps {len(upd['steps'])}/{upd['n_steps_planned']} "
                    f"subset NLL {upd['subset_nll_before']:.4f} -> {upd['subset_nll_after']:.4f} "
                    f"peak {row['peak_train_cuda_gib']} GiB train {row['train_sec'] / 60:.1f} min")
            with (out / "p5_rounds.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        del policy, opt, lora_params
        torch.cuda.empty_cache()
    log("ALL DONE")


if __name__ == "__main__":
    main()
