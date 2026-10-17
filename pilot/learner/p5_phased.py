"""P5 A2⁺ 미니 실행 — 단계 분리 버전 (수집과 학습을 라운드마다 별도 프로세스로).

배경: 한 프로세스에서 학습 뒤 환경 작업자를 다시 만들 때 Windows에서 충돌이 두 번 났다
(sapien 0x80000003, python310 0xc0000005, 둘 다 라운드 1 수집 시작). 작은 규모(6 에피소드)에서는
재현되지 않았다. 원인을 찾는 대신 단계마다 깨끗한 프로세스를 쓰고 상태를 디스크로 넘긴다.
라운드 단위로 이어서 할 수 있다.

상태 (seed마다 state_dir/seed{s}/):
  adapter/        PEFT LoRA 어댑터 (라운드 0 전에 학습 시드로 초기화해 저장)
  optimizer.pt    AdamW 상태
  round{r}_traj.npz  수집한 성공 궤적 (학습 뒤 지움)
  done.json       끝난 라운드 기록

학습 내용(U-prop, 토큰 평균 NLL, 클리핑 10, AdamW lr 1e-4, 미니배치 분할 누적)과 수집(64 에피소드, 온도 1.0, K0)은
p5_a2p_mini.py와 같다. KL은 기록만 한다(D-004).

사용: python p5_phased.py run <W1 ckpt> <out_dir> --seeds 0 1 --rounds 10
      (내부) python p5_phased.py collect|train ...
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))


def _load(ckpt, seed, adapter_dir, lr):
    import p5_a2p_mini as m
    import torch
    from types import SimpleNamespace
    from simpler_env.policies.openvla import openvla_train as ot
    from prismatic.extern.hf import modeling_prismatic as mp
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
    has_adapter = (adapter_dir / "adapter_config.json").exists()
    args = SimpleNamespace(vla_path=ckpt, vla_unnorm_key="bridge_orig",
                           vla_load_path=str(adapter_dir) if has_adapter else "", vla_lora_rank=32,
                           vla_lr=lr, vla_vhlr=3e-3, vla_optim_beta1=0.9, vla_optim_beta2=0.999,
                           vla_temperature=1.0, vla_temperature_eval=0.6)
    if not has_adapter:
        torch.manual_seed(seed)  # LoRA A 초기화 시드 = 학습 시드 (C2)
    policy = ot.OpenVLAPolicy(args, device_id=0)
    if not has_adapter:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        policy.vla.save_pretrained(str(adapter_dir))
    lora_params = [p for n, p in policy.vla.named_parameters() if "lora_" in n and p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr, betas=(0.9, 0.999))
    return policy, lora_params, opt


def phase_collect(a):
    import numpy as np
    import p5_a2p_mini as m
    import batched_eval as be
    sd = Path(a.state_dir)
    policy, _lp, _opt = _load(a.ckpt, a.seed, sd / "adapter", a.lr)
    unnorm = policy.vla.get_action_stats("bridge_orig")
    log = lambda s: print(s, flush=True)  # noqa: E731
    t0 = time.time()
    trajs, recs, retries = m.collect(policy, lambda: be.EnvPool(a.envs, unnorm, scale_seed=a.seed),
                                     a.seed, a.round, a.episodes, a.topk, log)
    np.savez(sd / f"round{a.round}_traj.npz",
             obs=np.stack([t["obs"] for t in trajs]) if trajs else np.zeros((0, 80, 480, 640, 3), np.uint8),
             tok=np.stack([t["tok"] for t in trajs]) if trajs else np.zeros((0, 80, 7), np.int64),
             instr=np.array([t["instruction"] for t in trajs]))
    (sd / f"round{a.round}_collect.json").write_text(json.dumps(dict(
        recs=recs, retries=retries, collect_sec=round(time.time() - t0, 1))), encoding="utf-8")
    log(f"[collect seed {a.seed} round {a.round}] successes {len(trajs)}/{len(recs)} in {(time.time() - t0) / 60:.1f} min")


def phase_train(a):
    import numpy as np
    import torch
    import p5_a2p_mini as m
    sd = Path(a.state_dir)
    d = np.load(sd / f"round{a.round}_traj.npz")
    trajs = [dict(obs=d["obs"][i], tok=d["tok"][i], instruction=str(d["instr"][i])) for i in range(len(d["obs"]))]
    policy, lora_params, opt = _load(a.ckpt, a.seed, sd / "adapter", a.lr)
    if (sd / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(sd / "optimizer.pt", map_location="cuda:0"))
    rng = np.random.default_rng(1000 * a.seed + a.round)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    upd = m.train_round(policy, lora_params, opt, trajs, a.mb, a.kl_max, rng, lambda s: print(s, flush=True))
    policy.vla.save_pretrained(str(sd / "adapter"))
    torch.save(opt.state_dict(), sd / "optimizer.pt")
    (sd / f"round{a.round}_train.json").write_text(json.dumps(dict(
        update=upd, train_sec=round(time.time() - t0, 1),
        peak_train_cuda_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))), encoding="utf-8")
    print(f"[train seed {a.seed} round {a.round}] steps {len(upd['steps'])}/{upd['n_steps_planned']} "
          f"subset NLL {upd['subset_nll_before']:.4f} -> {upd['subset_nll_after']:.4f}", flush=True)


def phase_run(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "p5_log.txt").open("a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    def sub(phase, seed, rnd, retries):
        cmd = [sys.executable, "-u", __file__, phase, a.ckpt, "--state-dir", str(out / f"seed{seed}"),
               "--seed", str(seed), "--round", str(rnd), "--episodes", str(a.episodes), "--envs", str(a.envs),
               "--topk", a.topk, "--mb", str(a.mb), "--lr", str(a.lr), "--kl-max", str(a.kl_max)]
        for attempt in range(retries + 1):
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            keep = [l for l in r.stdout.splitlines() if l.startswith(("[collect", "[train", "    step", "  [worker"))]
            for l in keep:
                log(l)
            if r.returncode == 0:
                return attempt
            log(f"  [{phase} failed] seed {seed} round {rnd} attempt {attempt + 1} exit {r.returncode}: "
                f"{r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
        raise RuntimeError(f"{phase} failed after {retries + 1} attempts (seed {seed} round {rnd})")

    for seed in a.seeds:
        sd = out / f"seed{seed}"
        sd.mkdir(parents=True, exist_ok=True)
        done = json.loads((sd / "done.json").read_text()) if (sd / "done.json").exists() else []
        for rnd in range(a.rounds):
            if rnd in done:
                continue
            t0 = time.time()
            c_fail = sub("collect", seed, rnd, retries=2) if not (sd / f"round{rnd}_collect.json").exists() else 0
            col = json.loads((sd / f"round{rnd}_collect.json").read_text())
            n_succ = sum(r["success"] for r in col["recs"])
            row = dict(seed=seed, round=rnd, episodes=len(col["recs"]), successes=n_succ,
                       collect_sec=col["collect_sec"], worker_retries=col["retries"], collect_process_failures=c_fail)
            if n_succ > 0:
                t_fail = sub("train", seed, rnd, retries=1)
                tr = json.loads((sd / f"round{rnd}_train.json").read_text())
                row.update(zero_success_round=False, train_process_failures=t_fail, **tr)
            else:
                row.update(zero_success_round=True, update=None)
            (sd / f"round{rnd}_traj.npz").unlink(missing_ok=True)
            with (out / "p5_rounds.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            with (out / "p5_episodes.jsonl").open("a", encoding="utf-8") as f:
                for r in col["recs"]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done.append(rnd)
            (sd / "done.json").write_text(json.dumps(done))
            log(f"[seed {seed} round {rnd}] successes {n_succ}/{row['episodes']}  round {(time.time() - t0) / 60:.1f} min")
    log("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "collect", "train"])
    ap.add_argument("ckpt")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--state-dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--round", type=int)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--envs", type=int, default=3)
    ap.add_argument("--topk", default="K0")
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-max", type=float, default=1e9)
    a = ap.parse_args()
    {"run": phase_run, "collect": phase_collect, "train": phase_train}[a.phase](a)
