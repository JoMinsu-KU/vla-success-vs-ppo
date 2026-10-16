"""P4 단위 시험 — 실제 OpenVLA가 필요한 항목 (C1·C2·C4). Windows 네이티브 rlvla_win, GPU 1장.

C1 행동 정규화: bridge_orig 통계로 256개 bin 전부를 역정규화해 실행 명령(그리퍼 포함)을 만들고,
   sft 통계와의 차이를 기록한다. 통제 연구는 모든 갈래가 bridge_orig를 쓴다.
C2 adapter 초기화: 병합된 W1 위에 원 PPO 레시피의 LoraConfig(rank 32, alpha 16, gaussian A·0 B)를 붙였을 때
   (i) 초기 로짓이 기본 모델과 같은지, (ii) 같은 초기화 시드면 A 행렬이 같고 다른 시드면 다른지, (iii) 학습 가능
   파라미터 목록이 시드와 무관하게 같은지.
C4 샘플링 분포 (결정적 비교): generate(output_scores=True, output_logits=True)로 같은 호출 안의 각 생성 위치에서
   샘플러가 쓴 처리된 분포(scores)와 학습 경로가 쓰는 256개 토큰 log-softmax(원 로짓)를 비교한다.
   (자기회귀 prefix는 그 호출에서 실제로 생성된 토큰으로 고정된다.)
     K0  (top_k 0, 온도 1): 두 분포가 수치 오차 안에서 같아야 한다.
     K50 (top_k 50, 온도 1): 상위 50 밖 질량만큼 달라야 한다 (원 코드의 불일치 재현). 그리고 대안 K50의
         절단·재정규화 log-prob(log π − log Σ_top50 π)가 샘플러 분포와 같아야 한다.
   왕복 검사: predict_action_batch가 돌려준 저장 log-prob와, 같은 토큰을 teacher forcing으로 다시 계산한
   evaluate_action log-prob가 같은지.

사용: python test_model_gpu.py <W1 체크포인트 폴더> <out_json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import p2_eval  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ckpt, out_json = sys.argv[1], Path(sys.argv[2])
dev = torch.device("cuda:0")
report = {}

# ---------------------------------------------------------------- C1
bins = np.linspace(-1, 1, 256)
centers = (bins[:-1] + bins[1:]) / 2.0
policy, bridge = p2_eval.load_policy(ckpt, "bridge_orig", 0.6, merge_lora=True)
norm = policy.vla.norm_stats
report["C1"] = {"available_unnorm_keys": sorted(norm.keys())}


def unnorm_all(stats):
    lo, hi = np.array(stats["q01"]), np.array(stats["q99"])
    mask = np.array(stats.get("mask", np.ones_like(lo, dtype=bool)))
    a = np.where(mask, 0.5 * (centers[:, None] + 1) * (hi - lo) + lo, centers[:, None])  # [255, 7]
    grip = 2.0 * (a[:, 6] > 0.5) - 1.0
    return a, grip


a_b, g_b = unnorm_all(norm["bridge_orig"]["action"])
w = p2_eval.WinSimplerWrapper.__new__(p2_eval.WinSimplerWrapper)
w.unnorm_state, w.bin_centers = norm["bridge_orig"]["action"], centers
tokens = torch.tensor([[32000 - 1 - i] * 7 for i in range(255)])  # dact = 32000 − token − 1 = i
cmd = w._process_action(tokens).numpy()
report["C1"]["wrapper_matches_reference_xyz_rot"] = bool(np.allclose(cmd[:, :6], a_b[:, :6], atol=1e-6))
report["C1"]["wrapper_matches_reference_gripper"] = bool(np.array_equal(cmd[:, 6], g_b))
if "sft" in norm:
    a_s, g_s = unnorm_all(norm["sft"]["action"])
    report["C1"]["bridge_vs_sft_max_abs_diff_per_dim"] = np.abs(a_b - a_s).max(0).round(5).tolist()
    report["C1"]["bridge_vs_sft_gripper_disagree_bins"] = int((g_b != g_s).sum())
print("[C1]", report["C1"], flush=True)

# ---------------------------------------------------------------- 관측 준비
pw = p2_eval.WinSimplerWrapper("PutOnPlateInScene25Main-v3", bridge, seed=0)
obs_list, instr_list = [], []
for k in range(6):
    o, ins, _ = pw.reset_with_episode("train", p2_eval.episode_id_for(5, k))
    obs_list.append(o)
    instr_list.append(ins)
pw.env.close()


def feats_of(i):
    return policy._preprocess_obs(dict(image=obs_list[i].to(dev), task_description=instr_list[i]))


# ---------------------------------------------------------------- C2
from peft import LoraConfig, get_peft_model  # noqa: E402

base = policy.vla
with torch.no_grad():
    f0 = feats_of(0)
    logits_base = base(input_ids=f0["input_ids"], attention_mask=f0["attention_mask"],
                       pixel_values=f0["pixel_values"]).logits[:, -1].float()


def lora_cfg():
    return LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, init_lora_weights="gaussian",
                      target_modules=["proj", "qkv", "fc1", "fc2", "q", "kv", "fc3", "q_proj", "k_proj", "v_proj",
                                      "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"])


def wrap(seed):
    torch.manual_seed(seed)
    m = get_peft_model(base, lora_cfg())
    with torch.no_grad():
        lg = m(input_ids=f0["input_ids"], attention_mask=f0["attention_mask"],
               pixel_values=f0["pixel_values"]).logits[:, -1].float()
    a_sum = sum(float(p.double().sum()) for n, p in m.named_parameters() if "lora_A" in n)
    names = sorted(n for n, p in m.named_parameters() if p.requires_grad)
    m.unload()  # LoRA 층 제거, 기본 모델 복원
    return lg, a_sum, names


lg1, a1, names1 = wrap(123)
lg1b, a1b, names1b = wrap(123)
lg2, a2, names2 = wrap(456)
report["C2"] = dict(
    init_logits_equal_base=bool(torch.equal(lg1, logits_base)),
    init_logits_max_abs_diff=float((lg1 - logits_base).abs().max()),
    same_seed_same_A=(a1 == a1b), different_seed_different_A=(a1 != a2),
    trainable_param_names_identical=(names1 == names2 == names1b), n_trainable_tensors=len(names1),
)
print("[C2]", report["C2"], flush=True)
policy.vla = base

# ---------------------------------------------------------------- C4
from prismatic.extern.hf.modeling_prismatic import AllowedTokensLogitsProcessor  # noqa: E402
from transformers import LogitsProcessorList  # noqa: E402

LO, HI = 32000 - 256, 32000


def c4_case(top_k, n_calls=6):
    gc = policy.vla.generation_config
    gc.top_k, gc.top_p = top_k, 1.0
    rows = []
    for i in range(n_calls):
        f = feats_of(i % len(obs_list))
        torch.manual_seed(1000 + i)
        with torch.no_grad():
            out = policy.vla.generate(f["input_ids"], attention_mask=f["attention_mask"],
                                      pixel_values=f["pixel_values"], max_new_tokens=7, do_sample=True,
                                      temperature=1.0, return_dict_in_generate=True, output_scores=True,
                                      output_logits=True,
                                      logits_processor=LogitsProcessorList([AllowedTokensLogitsProcessor()]))
        for t in range(7):
            raw = out.logits[t][0].float()
            proc = out.scores[t][0].float()
            p_sampler = torch.softmax(proc, -1)[LO:HI]                  # 샘플러가 실제로 쓴 분포 (256 bin)
            lp_train = torch.log_softmax(raw[LO:HI], -1)                # 학습 경로 분포 (원 코드·K0)
            outside = float(p_sampler.sum().item())
            p_s = p_sampler / p_sampler.sum()
            mass_out_top = 1.0 - float(lp_train.exp().topk(50).values.sum()) if top_k else 0.0
            tv_train = 0.5 * float((p_s - lp_train.exp()).abs().sum())
            # 대안 K50: 상위 50으로 절단·재정규화한 학습 log-prob
            keep = torch.zeros(256, dtype=torch.bool, device=raw.device)
            keep[lp_train.topk(50).indices] = True
            lp_trunc = torch.where(keep, lp_train - torch.logsumexp(lp_train[keep], 0), torch.full_like(lp_train, -1e9))
            tv_trunc = 0.5 * float((p_s - lp_trunc.exp()).abs().sum())
            rows.append(dict(call=i, pos=t, sampler_mass_in_256=outside, tv_sampler_vs_train=tv_train,
                             tv_sampler_vs_trunc50=tv_trunc, mass_outside_top50=mass_out_top,
                             token_in_top50=bool(keep[int(out.sequences[0, -7 + t]) - LO])))
    return rows


def summarize(rows):
    return dict(n_positions=len(rows),
                max_tv_sampler_vs_train=max(r["tv_sampler_vs_train"] for r in rows),
                mean_tv_sampler_vs_train=float(np.mean([r["tv_sampler_vs_train"] for r in rows])),
                max_tv_sampler_vs_trunc50=max(r["tv_sampler_vs_trunc50"] for r in rows),
                max_mass_outside_top50=max(r["mass_outside_top50"] for r in rows),
                mean_mass_outside_top50=float(np.mean([r["mass_outside_top50"] for r in rows])),
                all_sampled_tokens_in_top50=all(r["token_in_top50"] for r in rows),
                min_sampler_mass_in_256=min(r["sampler_mass_in_256"] for r in rows))


report["C4_K0"] = summarize(c4_case(0))
report["C4_K50"] = summarize(c4_case(50))
print("[C4 K0]", report["C4_K0"], flush=True)
print("[C4 K50]", report["C4_K50"], flush=True)

# 합성 오류 대조: 상위 50 밖에 질량 0.2를 둔 합성 로짓에서 K50 샘플러는 그 질량을 잘라야 한다
from transformers.generation.logits_process import TopKLogitsWarper  # noqa: E402

lg = torch.full((1, 32064), -1e9)
lg[0, LO:LO + 50] = 0.0
lg[0, LO + 50:HI] = float(np.log(0.2 / 0.8 * 50 / 206))  # 바깥 206개 합 질량 0.2
p_full = torch.softmax(lg[0, LO:HI], -1)
p_k50 = torch.softmax(TopKLogitsWarper(50)(None, lg.clone())[0, LO:HI], -1)
report["C4_synthetic"] = dict(outside_mass_full=float(p_full[50:].sum()), outside_mass_after_top50=float(p_k50[50:].sum()),
                              detects_truncation=bool(0.5 * (p_full - p_k50).abs().sum() > 0.15))
print("[C4 synthetic]", report["C4_synthetic"], flush=True)

# 왕복: 저장 log-prob (predict_action_batch) 대 teacher forcing 재계산 (evaluate_action)
gc = policy.vla.generation_config
gc.top_k = 0
rt = []
for i in range(len(obs_list)):
    x = dict(image=obs_list[i].to(dev), task_description=instr_list[i])
    torch.manual_seed(2000 + i)
    with torch.no_grad():
        _v, act, lp_store = policy.get_action(x, deterministic=False)
        lp_re, _ent, _v2 = policy.evaluate_actions(x, act)
    rt.append(float((lp_store.float() - lp_re.float()).abs().max()))
report["C4_roundtrip"] = dict(n=len(rt), max_abs_diff=max(rt), diffs=rt)
print("[C4 roundtrip]", report["C4_roundtrip"], flush=True)

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("SAVED", out_json, flush=True)
