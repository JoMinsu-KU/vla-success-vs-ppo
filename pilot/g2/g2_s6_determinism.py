"""G2 S6: C11 결정성과 비용 (명세 3.2절 C11, 6차 검토 4·5번).

엄격 모드(use_deterministic_algorithms(True, warn_only=False), CUBLAS_WORKSPACE_CONFIG=:4096:8, TF32 끔)에서
attention 경로 두 가지를 따로 실행한다 (프로세스마다 하나):
  --path flash_det : LLM attention = flash_attention_2, flash-attn 역전파를 deterministic=True로 호출
                     (transformers 4.40.1 llama가 쓰는 flash_attn_func·flash_attn_varlen_func에 deterministic=True를 넘김)
  --path math      : LLM attention = sdpa, math 커널만 켬
  두 경로 모두 비전 타워(timm)의 F.scaled_dot_product_attention은 numerics 설정으로 math 커널.
시험:
  1. 64 환경 1라운드 수집(온도 1.0, K0, S5와 같은 episode_id)
  2. C9 첫 optimizer 스텝: 원 코드형 두 번(A/A) + 통제형 → 결정적 설정에서 A/A가 0인지, C9 차이 크기
  3. A/A: (i) 새 LoRA·optimizer 시작점, (ii) 한 라운드 업데이트(다른 미니배치 순서) 뒤 Adam 모멘트 시작점에서
     같은 통제 PPO 라운드 업데이트(C10 재계산 포함, 32스텝)를 두 번 → LoRA·value head 비트 단위 동일?
  4. 비용: 라운드 업데이트 시간 (S5 기본 설정 값과 비교는 보고서에서)
엄격 모드 오류가 나면 연산 이름·위치를 기록하고 멈춘다.
사용: python g2_s6_determinism.py <W1 ckpt> <out_dir> --path flash_det|math
"""
import argparse
import functools
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_gpu as G  # noqa: E402,F401  (learner 경로 등록)
import numerics  # noqa: E402

numerics.enable_deterministic(strict=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import openvla_adapter as oa  # noqa: E402
import ppo  # noqa: E402
from g2_s5_scale import gvec, loss_and_backward, pvec, rel  # noqa: E402


def patch_flash_deterministic():
    import flash_attn
    from transformers.models.llama import modeling_llama as ml
    done = []
    for name in ("flash_attn_func", "flash_attn_varlen_func"):
        if hasattr(ml, name):
            f = getattr(ml, name)
            setattr(ml, name, functools.partial(f, deterministic=True))
            done.append(name)
    return dict(flash_attn=flash_attn.__version__, patched=done)


def vec(ps):
    return torch.cat([p.detach().double().reshape(-1).cpu() for p in ps])


def aa(ad, batch, start, cfg, label, log):
    outs = []
    for _ in range(2):
        ppo.restore(ad, start)
        torch.cuda.synchronize()
        t0 = time.time()
        r = ppo.ppo_update(ad, batch, cfg)
        torch.cuda.synchronize()
        outs.append((vec(ad.params_policy), vec(ad.params_value), time.time() - t0, r))
    p0 = torch.cat([q.double().reshape(-1).cpu() for q in start["policy"]])
    res = dict(lora_bitwise_equal=bool(torch.equal(outs[0][0], outs[1][0])),
               value_bitwise_equal=bool(torch.equal(outs[0][1], outs[1][1])),
               lora_max_abs_diff=float((outs[0][0] - outs[1][0]).abs().max()),
               lora_diff_norm_over_update_norm=float((outs[0][0] - outs[1][0]).norm() / max(1e-30, float((outs[0][0] - p0).norm()))),
               lora_frac_elements_different=float((outs[0][0] != outs[1][0]).double().mean()),
               update_sec=[round(outs[0][2], 1), round(outs[1][2], 1)], n_steps=outs[0][3]["n_steps"],
               first_minibatch=outs[0][3]["first_minibatch"], all_finite=all(s["finite"] for s in outs[0][3]["steps"]))
    log(f"[A/A {label}] {json.dumps(res)}")
    return res, outs[0][3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--path", required=True, choices=["flash_det", "math"])
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--smoke-steps", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(args=vars(a))
    t_all = time.time()

    def log(m):
        print(m, flush=True)

    def save():
        res["total_min"] = round((time.time() - t_all) / 60, 1)
        res["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        (out / f"g2_s6_{a.path}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    try:
        if a.path == "flash_det":
            res["flash_patch"] = patch_flash_deterministic()
            attn = "flash_attention_2"
        else:
            attn = "sdpa"
        policy = G.build_policy(a.ckpt, "bridge_orig", attn=attn, seed=0)
        G.set_top_k(policy, "K0")
        res["numerics"] = numerics.numerics_report(policy.vla)
        log(f"numerics {json.dumps(res['numerics'])}")
        unnorm = policy.vla.get_action_stats("bridge_orig")
        env = G.GpuEnv("PutOnPlateInScene25Main-v3", unnorm, a.n_envs, seed=0)
        eids = [G.episode_id_for("g2s5-seed0-round0", k) for k in range(a.n_envs)]
        tm = {}
        r = G.rollout(policy, env, "train", eids, deterministic=False, rng_seed=0, record_obs=True, timing=tm)
        env.close()
        torch.cuda.empty_cache()
        res["collect"] = dict({k: round(v, 1) for k, v in tm.items()}, Y=int(r["Y"].sum()))
        log(f"[collect] {res['collect']}")
        batch = G.to_roundbatch(r)
        ad = oa.OpenVLAAdapter(policy, r["obs"], r["instr"], r["actions"])
        n = batch.T * batch.N
        params = list(ad.params_policy) + list(ad.params_value)
        snap0 = ppo.snapshot(ad)

        # C9 첫 스텝 (결정적 설정): 원 코드형 A/A + 통제형
        old_lp = batch.logprob_gen.reshape(-1).astype(np.float32)
        old_v = batch.value_gen.reshape(-1).astype(np.float32)
        adv, ret, _ = ppo.advantages_and_returns(batch, batch.value_gen, "terminal", "gae")
        plan = ppo.plan_steps(ppo.minibatch_order(n, 0), 8, 20, controlled=False)
        _s, mbs0, div0 = plan[0]
        Gs, Ls = {}, {}
        for key in ("original", "original_repeat", "controlled"):
            mode = "controlled" if key == "controlled" else "original"
            ppo.restore(ad, snap0)
            ad.opt_policy.zero_grad(set_to_none=True)
            ad.opt_value.zero_grad(set_to_none=True)
            ad.train_mode()
            Ls[key] = loss_and_backward(ad, mbs0, div0, mode, old_lp, old_v, adv.reshape(-1), ret.reshape(-1), a.chunk)
            Gs[key] = gvec(params).cpu()
        res["C9_first_step_deterministic"] = dict(
            aa_loss_equal=Ls["original"] == Ls["original_repeat"], aa_grad_bitwise_equal=bool(torch.equal(Gs["original"], Gs["original_repeat"])),
            c9_loss_rel_diff=abs(Ls["original"] - Ls["controlled"]) / max(1e-30, abs(Ls["original"])),
            c9_grad_rel_diff=rel(Gs["controlled"], Gs["original"]),
            c9_grad_frac_elements_different=float((Gs["controlled"] != Gs["original"]).double().mean()))
        log(f"[C9 det] {res['C9_first_step_deterministic']}")
        del Gs
        ad.opt_policy.zero_grad(set_to_none=True)
        ad.opt_value.zero_grad(set_to_none=True)
        save()

        cfg = ppo.PPOConfig(order_seed=7, chunk=a.chunk, recompute_old=True, max_steps=a.smoke_steps or None)
        res["aa_fresh"], upd = aa(ad, batch, snap0, cfg, "fresh W1 + new LoRA/Adam", log)
        res["c10_audit_W1"] = upd["c10_audit"]
        save()
        ppo.restore(ad, snap0)
        ppo.ppo_update(ad, batch, ppo.PPOConfig(order_seed=11, chunk=a.chunk, recompute_old=True, max_steps=a.smoke_steps or None))
        snap1 = ppo.snapshot(ad)
        res["aa_after_one_round"], _ = aa(ad, batch, snap1, cfg, "after one round (Adam moments)", log)
        res["strict_error"] = None
    except Exception as e:
        res["strict_error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        log(f"STRICT/RUN ERROR {res['strict_error']}")
    save()
    log("SAVED")


if __name__ == "__main__":
    main()
