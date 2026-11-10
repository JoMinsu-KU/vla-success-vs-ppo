"""E2-a 누적 기울기 진단 1단위 (계보 s, 분기 지점 p). 규칙: study/E2a_design.md (실행 전 고정).

사용: python study/e2a.py --job-id e… --lineage-ckpt <W>/runs/<계보>/ckpt/r(p−1).pt --seed s --point p
      [--sizes 1,2,4,8] [--ood Position,EEPose,PositionChangeTo] [--smoke]
출력: <W>/blinded/e2a/<job>.json (결과), <W>/runs/e2a/<job>/theta1_<규칙>.pt (크기 1의 π_new), <W>/public/e2a/<job>.json (상태·검사)
임시: <W>/scratch/e2a/<job>/ (배치 관측 memmap, 끝나면 지운다)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--job-id", required=True)
ap.add_argument("--lineage-ckpt", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--point", type=int, required=True)
ap.add_argument("--sizes", default="1,2,4,8")
ap.add_argument("--ood", default="Position,EEPose,PositionChangeTo")
ap.add_argument("--n-diag", type=int, default=64)
ap.add_argument("--n-ll", type=int, default=256)
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--smoke", action="store_true", help="시험용: 배치당 전이를 앞 몇 시점만 써서 빨리 끝낸다 (결과는 쓰지 않는다)")
a = ap.parse_args()

W = C.work_dir()
C.ensure_dirs(W)
OUT = W / "blinded" / "e2a" / f"{a.job_id}.json"
PUB = W / "public" / "e2a" / f"{a.job_id}.json"
TMP = W / "scratch" / "e2a" / a.job_id
THETA = W / "runs" / "e2a" / a.job_id
for d in (OUT.parent, PUB.parent, TMP, THETA):
    d.mkdir(parents=True, exist_ok=True)
SIZES = [int(x) for x in a.sizes.split(",")]
NB = max(SIZES)
s, p = a.seed, a.point

import numerics  # noqa: E402

numerics.enable_deterministic(strict=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import core  # noqa: E402
import e2  # noqa: E402
import g2_gpu as G  # noqa: E402
import g2_learn as L  # noqa: E402
import openvla_adapter as oa  # noqa: E402
import ppo  # noqa: E402

T_USE = 4 if a.smoke else 80  # 스모크: 배치당 앞 4시점만 기울기에 사용


def pub(status, **kw):
    C.write_json(PUB, dict(job_id=a.job_id, status=status, seed=s, point=p, updated=C.now(), **kw))


def bseed(b: int) -> int:
    return 900_000_000 + 1_000_000 * s + 1_000 * p + b


pub("running")
t_start = time.time()
policy = G.build_policy(C.w1_ckpt(), C.CFG["unnorm_key"], attn="sdpa", seed=s)
G.set_top_k(policy, C.CFG["top_k"])
ex = L.load_trainable(Path(a.lineage_ckpt), policy, load_optim=True)
assert int(ex["rounds_done"]) == p - 1, (ex["rounds_done"], p)
unnorm = policy.vla.get_action_stats(C.CFG["unnorm_key"])
dev = torch.device("cuda:0")


# ---------------------------------------------------------------- 1. 배치 수집
def collect(env, split, tag, rng_seed, keep_obs_path=None):
    eids = [G.episode_id_for(tag, k) for k in range(64)]
    rec = G.rollout(policy, env, split, eids, deterministic=False, rng_seed=rng_seed, record_obs=True)
    obs = rec.pop("obs")
    if keep_obs_path is not None:
        mm = np.lib.format.open_memmap(keep_obs_path, mode="w+", dtype=np.uint8, shape=obs.shape)
        mm[:] = obs
        mm.flush()
        del mm
    return rec, obs


env = G.GpuEnv(C.CFG["train_env"], unnorm, 64, seed=s)
batches = []
for b in range(1, NB + 1):
    tag, rs = (f"e1-seed{s}-round{p - 1}", 100_000 * s + p - 1) if b == 1 else (f"e2a-seed{s}-p{p}-b{b}", bseed(b))
    rec, obs = collect(env, C.CFG["train_obj_set"], tag, rs, TMP / f"obs_b{b}.npy")
    del obs
    batches.append(rec)
    print(f"[collect] b{b} done {time.time() - t_start:.0f}s", flush=True)
rec_d, obs_d = collect(env, C.CFG["train_obj_set"], f"e2a-diag-seed{s}-p{p}", bseed(99))
env.close()


def sample_states(obs, rec, n, seed):
    T, N = rec["actions"].shape[:2]
    rng = np.random.default_rng(seed)
    flat = rng.choice(T * N, size=min(n, T * N), replace=False)
    t, i = flat // N, flat % N
    return dict(obs=obs[t, i].copy(), instr=[rec["instr"][k] for k in i], act=rec["actions"][t, i].copy())


diag = {"train": sample_states(obs_d, rec_d, a.n_diag, 10_000 + p)}
Yd = rec_d["Y"].astype(bool)
ll_sets = {}
for name, sel in (("success", Yd), ("failure", ~Yd)):
    T, N = rec_d["actions"].shape[:2]
    idx = np.flatnonzero(np.tile(sel, T))  # 전이 t·N + i
    rng = np.random.default_rng(20_000 + p + (0 if name == "success" else 1))
    pick = rng.choice(idx, size=min(a.n_ll, len(idx)), replace=False) if len(idx) else np.array([], int)
    t, i = pick // N, pick % N
    ll_sets[name] = dict(obs=obs_d[t, i].copy(), instr=[rec_d["instr"][k] for k in i], act=rec_d["actions"][t, i].copy())
del obs_d
for k, cond in enumerate([c for c in a.ood.split(",") if c]):
    cfg_c = [c for c in C.CFG["eval_conditions"] if c[0] == cond][0]
    env = G.GpuEnv(cfg_c[1], unnorm, 64, seed=s)
    r_o, o_o = collect(env, cfg_c[2], f"e2a-ood-{cond}-seed{s}-p{p}", bseed(101 + k))
    env.close()
    diag[cond] = sample_states(o_o, r_o, a.n_diag, 10_100 + p + k)
    del o_o
print(f"[collect] diag done {time.time() - t_start:.0f}s", flush=True)

# ---------------------------------------------------------------- 2. θ_old 재계산 (teacher forcing, train 모드)
params_pol = list(policy.params_vla)
params_vh = list(policy.params_vh)
params_all = params_pol + params_vh
n_pol = sum(q.numel() for q in params_pol)
AD0 = oa.OpenVLAAdapter(policy, None, [], np.zeros((1, 1, 7), np.int64))
snap = ppo.snapshot(AD0)


def restore_old():
    """θ_old (계보 체크포인트의 파라미터·두 Adam 상태)로 되돌린다. 모든 기울기는 θ_old에서 잰다."""
    ppo.restore(AD0, snap)


def adapter_of(b):
    rec = batches[b - 1]
    obs = np.load(TMP / f"obs_b{b}.npy", mmap_mode="r")
    return oa.OpenVLAAdapter(policy, obs, rec["instr"], rec["actions"])


def tidx():  # 기울기에 쓰는 전이 (t < T_USE), 전이 인덱스 t·64 + i
    return np.arange(T_USE * 64)


old = []
for b in range(1, NB + 1):
    ad = adapter_of(b)
    ad.train_mode()
    n = 80 * 64
    lp_all, v_all = np.zeros(n, np.float32), np.zeros(n, np.float32)
    with torch.no_grad():
        for j in range(0, n, a.chunk):
            idx = np.arange(j, min(n, j + a.chunk))
            lp, v = ad.evaluate(idx)
            lp_all[idx] = lp[:, 0].float().cpu().numpy()
            v_all[idx] = v[:, 0].float().cpu().numpy()
    old.append((lp_all.reshape(80, 64), v_all.reshape(80, 64)))


def roundbatch(recs):
    cat = lambda k: np.concatenate([r[k] for r in recs], axis=1)  # noqa: E731
    return ppo.RoundBatch(actions=cat("actions"), logprob_gen=cat("lp_gen"), value_gen=cat("v_gen"), success=cat("success"),
                          grasp_latch=cat("grasp_latch"), consec_latch=cat("consec_latch"))


raw, rets, Ys, wbar = [], [], [], []
for b in range(1, NB + 1):
    rb = roundbatch([batches[b - 1]])
    _adv, ret, _ = ppo.advantages_and_returns(rb, old[b - 1][1], "terminal", "gae")
    raw.append(ret[:, :, 0] - old[b - 1][1])
    rets.append(ret[:, :, 0])
    Ys.append(np.broadcast_to(rb.Y().astype(np.float32)[None, :], (80, 64)))
    wbar.append(np.clip(np.exp(old[b - 1][0] - batches[b - 1]["lp_gen"]), 0.5, 2.0).astype(np.float32))


# ---------------------------------------------------------------- 3. 배치별 기울기 합 (한 번 순전파, 여러 번 역전파)
def flat_grads(loss, retain):
    gs = torch.autograd.grad(loss, params_all, retain_graph=retain, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(q)).reshape(-1).float() for g, q in zip(gs, params_all)])


def batch_sums(b, weights: dict, with_value: bool):
    """weights: 이름 → [80,64] 전이 가중치. 반환 이름 → Σ w·∇log π (+ "V": Σ ∇ℓ_V). 항상 θ_old에서 계산."""
    restore_old()
    ad = adapter_of(b)
    ad.train_mode()
    acc = {k: torch.zeros(sum(q.numel() for q in params_all), device=dev) for k in list(weights) + (["V"] if with_value else [])}
    ids = tidx()
    vp_all, r_all = old[b - 1][1].reshape(-1), rets[b - 1].reshape(-1)
    for j in range(0, len(ids), a.chunk):
        idx = ids[j:j + a.chunk]
        lp, v = ad.evaluate(idx)
        losses = []
        for k, w in weights.items():
            wt = torch.as_tensor(w.reshape(-1)[idx], device=dev).view(-1, 1)
            losses.append((k, (wt * lp).sum()))
        if with_value:
            vp = torch.as_tensor(vp_all[idx], device=dev).view(-1, 1)
            r = torch.as_tensor(r_all[idx], device=dev).view(-1, 1)
            losses.append(("V", core.value_loss_original(v, vp, r) * len(idx)))
        for q, (k, l_) in enumerate(losses):
            acc[k] += flat_grads(l_, retain=q < len(losses) - 1)
    return acc


ones = np.ones((80, 64), np.float32)
cum, at = None, {}
for b in range(1, NB + 1):
    w = {"S1": ones, "SA": raw[b - 1], "SY": Ys[b - 1]}
    if b == 1:
        w.update({"wS1": wbar[0], "wSA": wbar[0] * raw[0], "wSY": wbar[0] * Ys[0]})
    sums = batch_sums(b, w, with_value=True)
    if b == 1:
        at["c10"] = {k: sums.pop(k) for k in ("wS1", "wSA", "wSY")}
    cum = sums if cum is None else {k: cum[k] + sums[k] for k in cum}
    if b in SIZES:
        at[b] = {k: v.clone() for k, v in cum.items()}
    del sums
    print(f"[grad] b{b} {time.time() - t_start:.0f}s", flush=True)

# ---------------------------------------------------------------- 4. 규칙별 기울기
RULES = e2.RULES  # O± outcome, O⁺ outcome_masked, C± gae, C⁺ gae_masked
PC, VC = core.POLICY_COEF, core.VALUE_COEF


def official_adv(m, rule):
    rb = roundbatch(batches[:m])
    v = np.concatenate([o[1] for o in old[:m]], axis=1)
    adv = ppo.advantages_and_returns(rb, v, "terminal", rule)[0][:, :, 0]
    return adv[:T_USE]  # 정규화는 전체 버퍼(80시점)에서, 기울기는 쓰는 전이만


def affine_fit(adv, u):
    A = np.stack([u.reshape(-1).astype(np.float64), np.ones(u.size)], 1)
    coef, *_ = np.linalg.lstsq(A, adv.reshape(-1).astype(np.float64), rcond=None)
    resid = float(np.abs(A @ coef - adv.reshape(-1)).max())
    if resid > 1e-4:
        raise SystemExit(f"affine fit residual {resid} (규칙이 1차식이 아니다)")
    return float(coef[0]), float(coef[1]), resid


def policy_grad(m, rule_name, S, corrected=False):
    rule = RULES[rule_name]
    adv = official_adv(m, rule)
    n = T_USE * 64 * m
    if rule == "gae_masked":
        g = torch.zeros_like(S["S1"])
        for b in range(1, m + 1):
            wb = adv[:, 64 * (b - 1):64 * b]
            if corrected:
                wb = wb * wbar[b - 1][:T_USE]
            full = np.zeros((80, 64), np.float32)
            full[:T_USE] = wb
            g += batch_sums(b, {"g": full}, with_value=False)["g"]
        return -(PC / n) * g, None
    u = np.concatenate([raw[b][:T_USE] for b in range(m)], 1) if rule == "gae" else np.concatenate([Ys[b][:T_USE] for b in range(m)], 1)
    al, be, resid = affine_fit(adv, u)
    key = "SA" if rule == "gae" else "SY"
    if corrected:
        return -(PC / n) * (al * at["c10"]["w" + key] + be * at["c10"]["wS1"]), resid
    return -(PC / n) * (al * S[key] + be * S["S1"]), resid


def cos_ratio(x, y):
    return e2._cos_ratio(x.detach().cpu(), y.detach().cpu())


def one_step(g_total):
    """계보 Adam 상태에서 한 스텝 (클리핑 10 → value → policy). Δθ와 그 상태를 반환하고 되돌리지 않는다."""
    restore_old()
    off = 0
    for q in params_all:
        q.grad = g_total[off:off + q.numel()].view_as(q).to(q.dtype).clone()
        off += q.numel()
    before = torch.cat([q.detach().reshape(-1).float() for q in params_all])
    torch.nn.utils.clip_grad_norm_(params_all, C.CFG["grad_clip"])
    policy.vh_optimizer.step()
    policy.vla_optimizer.step()
    return torch.cat([q.detach().reshape(-1).float() for q in params_all]) - before


@torch.no_grad()
def dists(st):
    out = []
    for j in range(0, len(st["act"]), 16):
        x = dict(image=torch.from_numpy(st["obs"][j:j + 16]).to(dev), task_description=st["instr"][j:j + 16])
        lp, _v, d, _p = L.tf_full(policy, x, torch.from_numpy(st["act"][j:j + 16]).to(dev))
        out.append((lp[:, 0].float(), d.float()))
    return torch.cat([o[0] for o in out]) if out else torch.zeros(0, device=dev), torch.cat([o[1] for o in out]) if out else None


def policy_snapshot_stats():
    policy.prep_training()  # KL 탐침과 같은 모드 (train.py: ad.train_mode() 뒤 probe.set_ref)
    r = {}
    for k, st in diag.items():
        _lp, d = dists(st)
        r[k] = d
    ll = {k: dists(st)[0] for k, st in ll_sets.items()}
    return r, ll


def change_stats(ref, ref_ll):
    now, now_ll = policy_snapshot_stats()
    out = {}
    for k in ref:
        pr = ref[k].exp()
        kl = float((pr * (ref[k] - now[k])).sum(-1).sum(-1).mean())
        ent0 = float(-(pr * ref[k]).sum(-1).mean())
        ent1 = float(-(now[k].exp() * now[k]).sum(-1).mean())
        out[k] = dict(kl_old_new=kl, entropy_old=ent0, entropy_new=ent1)
    ood = [k for k in out if k != "train"]
    if ood:
        out["ood_mean_kl"] = float(np.mean([out[k]["kl_old_new"] for k in ood]))
    for k in ref_ll:
        out[f"ll_{k}"] = dict(n=int(ref_ll[k].numel()), delta_mean=float((now_ll[k] - ref_ll[k]).mean()) if ref_ll[k].numel() else None)
    return out


restore_old()
ref_d, ref_ll = policy_snapshot_stats()

res = dict(job_id=a.job_id, seed=s, point=p, sizes=SIZES, smoke=a.smoke, t_use=T_USE,
           batches=[dict(b=b, Y_mean=float(batches[b - 1]["Y"].mean())) for b in range(1, NB + 1)],
           diag_Y_mean=float(rec_d["Y"].mean()), wbar=dict(mean=float(wbar[0].mean()), frac_clipped=float(((wbar[0] == 0.5) | (wbar[0] == 2.0)).mean())),
           by_size={}, c10_size1={}, checks={})

# 분해 검사: 크기 1의 C± 정책 기울기를 직접 계산 (공식 advantage를 가중치로)
adv1 = official_adv(1, "gae")
full = np.zeros((80, 64), np.float32)
full[:T_USE] = adv1
g_direct = -(PC / (T_USE * 64)) * batch_sums(1, {"g": full}, with_value=False)["g"]
g_dec, resid = policy_grad(1, "C±", at[1])
chk = cos_ratio(g_direct, g_dec)
rel = abs(float(g_dec.norm()) - float(g_direct.norm())) / max(float(g_direct.norm()), 1e-30)
ok = chk["cos"] is not None and chk["cos"] >= 1 - 1e-5 and rel <= 1e-3
res["checks"]["decomposition_C±_size1"] = dict(cos=chk["cos"], norm_rel_err=rel, affine_resid=resid, passed=bool(ok))
if not ok:
    C.write_json(OUT, res)
    pub("failed", decomposition_passed=False)
    raise SystemExit("분해 검사 실패")

for m in SIZES:
    S = at[m]
    n = T_USE * 64 * m
    gV = (VC / n) * S["V"]
    g_pol = {name: policy_grad(m, name, S)[0] for name in RULES}
    g_tot = {name: g_pol[name] + gV for name in RULES}
    dth = {name: one_step(g_tot[name]) for name in RULES}
    pc_n = {}
    stats = {}
    if m in (1, NB):
        for name in RULES:
            one_step(g_tot[name])
            stats[name] = change_stats(ref_d, ref_ll)
            if m == 1:
                torch.save({nm: q.detach().cpu() for nm, q in policy.vla.named_parameters() if q.requires_grad},
                           THETA / f"theta1_{name}.pt")
    # C⁺ₙ: C⁺ 정책 기울기를 C± 노름으로
    sc = float(g_pol["C±"][:n_pol].norm()) / max(float(g_pol["C⁺"][:n_pol].norm()), 1e-30)
    g_cn = g_pol["C⁺"] * sc + gV
    dth_cn = one_step(g_cn)
    if m in (1, NB):
        stats["C⁺ₙ"] = change_stats(ref_d, ref_ll)
    pc_n = dict(scale=sc, delta_theta_vs_Cpm=cos_ratio(dth["C±"], dth_cn))
    row = {}
    for r_, (pm, pl) in {"O": ("O±", "O⁺"), "C": ("C±", "C⁺")}.items():
        row[r_] = dict(policy_grad=cos_ratio(g_pol[pm][:n_pol], g_pol[pl][:n_pol]), total_grad=cos_ratio(g_tot[pm], g_tot[pl]),
                       delta_theta=cos_ratio(dth[pm], dth[pl]))
    res["by_size"][str(m)] = dict(rows=row, C_plus_n=pc_n, policy_change=stats,
                                  degenerate_batches=int(sum(batches[b]["Y"].min() == batches[b]["Y"].max() for b in range(m))))
    print(f"[size {m}] {time.time() - t_start:.0f}s", flush=True)

# C10 보정 민감도 (크기 1, 정책 기울기)
for name in RULES:
    gu = policy_grad(1, name, at[1])[0]
    gc = policy_grad(1, name, at[1], corrected=True)[0]
    res["c10_size1"][name] = cos_ratio(gu[:n_pol], gc[:n_pol])

restore_old()
res["gpu_h"] = round((time.time() - t_start) / 3600, 2)
C.write_json(OUT, res)
(OUT.parent / f"sha256_{a.job_id}.txt").write_text(C.sha256_file(OUT) + "\n")
shutil.rmtree(TMP, ignore_errors=True)
pub("done", decomposition_passed=True, gpu_h_total=res["gpu_h"])
print("E2A DONE", flush=True)
