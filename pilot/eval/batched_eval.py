"""환경 N개를 별도 프로세스로 돌리고 정책 추론을 배치로 묶는 평가·수집 실행기 (Windows 네이티브).

배경: 배치 1 추론은 CPU 디스패치가 병목이라(GPU 사용률 약 26%) 배치를 키워도 스텝 시간이 거의 늘지 않는다.
Windows에는 GPU 병렬 시뮬레이션이 없으므로, CPU 시뮬레이션 환경을 프로세스 N개로 나눠 돌리고
같은 스텝의 관측을 모아 한 번에 추론한다.

정책 쪽은 원 OpenVLAPolicy의 전처리(_preprocess_obs)와 predict_action_batch를 그대로 쓰고,
환경 쪽은 p2_eval.WinSimplerWrapper(원 SimlerWrapper 메서드)를 그대로 쓴다.
모든 에피소드는 80스텝으로 같으므로 N개씩 묶어(웨이브) 같은 스텝을 진행한다.

에피소드 사양(spec)은 dict: env_id, obj_set, episode_id, 그리고 기록에 그대로 붙일 임의의 필드.
"""
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

EPISODE_LEN = 80


# ---------------------------------------------------------------- 환경 작업자
def _worker(conn, unnorm_state, scale_seed):
    import os
    import sys
    # 작업자는 CUDA가 필요 없다 (CPU 시뮬레이션, 렌더링은 Vulkan). torch import만으로 CUDA 컨텍스트가
    # 생겨 작업자마다 GPU 메모리를 차지하므로 CUDA를 숨긴다. Vulkan 렌더링은 영향받지 않는다.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import p2_eval  # noqa: F401  (win_compat·환경 변수)
    import torch

    wrapper, cur_env = None, None
    while True:
        msg = conn.recv()
        cmd = msg[0]
        if cmd == "reset":
            _, env_id, obj_set, episode_id = msg
            if env_id != cur_env:
                if wrapper is not None:
                    wrapper.env.close()
                wrapper = p2_eval.WinSimplerWrapper(env_id, unnorm_state, scale_seed)
                cur_env = env_id
            obs, instr, _ = wrapper.reset_with_episode(obj_set, int(episode_id))
            u = wrapper.env.unwrapped
            ids = p2_eval.selected_ids(wrapper.env)
            conn.send((obs[0].numpy(), instr[0], ids))
        elif cmd == "step":
            tokens = torch.as_tensor(msg[1]).reshape(1, 7)
            obs, _r, trunc, info = wrapper.step(tokens)
            u = wrapper.env.unwrapped
            name = u.carrot_names[int(u.select_carrot_ids.reshape(-1)[0])]
            grasp_now = bool(u.agent.is_grasping(u.objs_carrot[name]).reshape(-1)[0].item())
            step_info = dict(
                success=bool(info["success"].reshape(-1)[0].item()),
                grasp_now=grasp_now,
                grasped_latch=bool(info["is_src_obj_grasped"].reshape(-1)[0].item()),
                consecutive_latch=bool(info["consecutive_grasp"].reshape(-1)[0].item()),
                truncated=bool(trunc.reshape(-1)[0].item()),
            )
            conn.send((obs[0].numpy(), step_info))
        elif cmd == "dims":
            _, env_id = msg
            if env_id != cur_env:
                if wrapper is not None:
                    wrapper.env.close()
                wrapper = p2_eval.WinSimplerWrapper(env_id, unnorm_state, scale_seed)
                cur_env = env_id
            u = wrapper.env.unwrapped
            conn.send(dict(n_pos=int(len(u.xyz_configs)), n_quat=int(len(u.quat_configs)),
                           n_carrot=len(u.carrot_names), n_plate=len(u.plate_names),
                           carrot_names=list(u.carrot_names)))
        elif cmd == "close":
            if wrapper is not None:
                wrapper.env.close()
            conn.send("closed")
            return


class EnvPool:
    def __init__(self, n, unnorm_state, scale_seed=0):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, unnorm_state, scale_seed), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def dims(self, env_id):
        self.conns[0].send(("dims", env_id))
        return self.conns[0].recv()

    def reset(self, specs):
        for c, s in zip(self.conns, specs):
            c.send(("reset", s["env_id"], s["obj_set"], s["episode_id"]))
        return [c.recv() for c in self.conns[:len(specs)]]

    def step(self, tokens):
        for c, t in zip(self.conns, tokens):
            c.send(("step", t))
        return [c.recv() for c in self.conns[:len(tokens)]]

    def close(self):
        for c in self.conns:
            try:
                c.send(("close",))
                c.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)


# ---------------------------------------------------------------- 정책
def set_top_k(policy, mode: str):
    """K50: transformers 기본 top_k 50 (원 코드형). K0: top-k 절단 해제 (v0.9 C4 기본안)."""
    gc = policy.vla.generation_config
    gc.top_k = 50 if mode == "K50" else 0
    gc.top_p = 1.0


def policy_step(policy, obs_list, instr_list, unnorm_key, temperature, device):
    import torch
    images = torch.from_numpy(np.stack(obs_list)).to(device)
    feats = policy._preprocess_obs(dict(image=images, task_description=list(instr_list)))
    with torch.no_grad():
        _v, actions, logprobs = policy.vla.predict_action_batch(
            **feats, unnorm_key=unnorm_key, do_sample=temperature > 0, temperature=temperature)
    return actions.cpu().numpy(), logprobs.float().cpu().numpy().reshape(-1)


def run_specs(policy, pool, specs, unnorm_key, temperature, rng_seed, on_episode, log=print):
    """specs를 풀 크기만큼씩 묶어 진행한다. 에피소드가 끝날 때마다 on_episode(spec, rec)를 부른다."""
    import torch
    device = torch.device("cuda:0")
    n = len(pool.conns)
    for w0 in range(0, len(specs), n):
        wave = specs[w0:w0 + n]
        t0 = time.time()
        torch.manual_seed(rng_seed + w0)
        resets = pool.reset(wave)
        obs = [r[0] for r in resets]
        instr = [r[1] for r in resets]
        traces = [dict(success=[], grasp_now=[], logprob=[]) for _ in wave]
        last = [None] * len(wave)
        t_pol = 0.0
        for t in range(EPISODE_LEN):
            tp = time.time()
            toks, lps = policy_step(policy, obs, instr, unnorm_key, temperature, device)
            t_pol += time.time() - tp
            outs = pool.step(list(toks))
            obs = [o[0] for o in outs]
            for i, (_o, si) in enumerate(outs):
                traces[i]["success"].append(si["success"])
                traces[i]["grasp_now"].append(si["grasp_now"])
                traces[i]["logprob"].append(round(float(lps[i]), 4))
                last[i] = si
        dt = time.time() - t0
        for i, s in enumerate(wave):
            si = last[i]
            assert si["truncated"], "episode did not truncate at step 80"
            rec = {k: v for k, v in s.items()}
            rec.update(resets[i][2])
            rec.update(
                instruction=instr[i],
                success=si["success"],                      # 원 평가 기준: 마지막 스텝
                grasped_latch=si["grasped_latch"],
                consecutive_latch=si["consecutive_latch"],
                success_and_grasp=si["success"] and si["grasped_latch"],
                success_any_step=any(traces[i]["success"]),
                success_trace="".join("1" if x else "0" for x in traces[i]["success"]),
                grasp_trace="".join("1" if x else "0" for x in traces[i]["grasp_now"]),
                logprob_sum=round(sum(traces[i]["logprob"]), 3),
                temperature=temperature,
                wave_sec=round(dt, 2), wave_policy_sec=round(t_pol, 2), wave_size=len(wave),
            )
            on_episode(s, rec)
        log(f"[wave {w0 // n + 1}/{(len(specs) + n - 1) // n}] {len(wave)} eps in {dt:.1f}s "
            f"(policy {t_pol:.1f}s) -> {dt / len(wave):.1f} s/ep, success {sum(r['success'] for r in [{'success': l['success']} for l in last])}/{len(wave)}")
