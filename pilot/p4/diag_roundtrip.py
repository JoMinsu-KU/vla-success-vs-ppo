"""P4 C4 왕복 불일치 진단: 저장 log-prob(generate, KV 캐시 증분 디코딩) 대 재계산(evaluate_action, teacher forcing).

원인 후보를 나눠 잰다.
  (a) 재토큰화: evaluate_action은 행동 토큰을 문자열로 decode한 뒤 프롬프트에 넣어 다시 tokenize한다.
      다시 나온 7개 토큰이 생성된 토큰과 같은가?
  (b) 수치 경로: 같은 토큰 열(직접 input_ids로 붙임)을 한 번에 forward했을 때의 위치별 로짓과
      generate의 위치별 로짓(output.logits)의 차이. bf16 log_softmax와 fp32 log_softmax 차이도 함께.
사용: python diag_roundtrip.py <ckpt> <out_json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import p2_eval  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from prismatic.extern.hf.modeling_prismatic import AllowedTokensLogitsProcessor  # noqa: E402
from transformers import LogitsProcessorList  # noqa: E402

ckpt, out_json = sys.argv[1], Path(sys.argv[2])
dev = torch.device("cuda:0")
policy, unnorm = p2_eval.load_policy(ckpt, "bridge_orig", 0.6, merge_lora=True)
policy.vla.generation_config.top_k = 0
w = p2_eval.WinSimplerWrapper("PutOnPlateInScene25Main-v3", unnorm, seed=0)
LO, HI = 32000 - 256, 32000
rows = []
for i in range(12):
    obs, instr, _ = w.reset_with_episode("train", p2_eval.episode_id_for(7, i))
    x = dict(image=obs.to(dev), task_description=instr)
    f = policy._preprocess_obs(x)
    torch.manual_seed(500 + i)
    with torch.no_grad():
        out = policy.vla.generate(f["input_ids"], attention_mask=f["attention_mask"], pixel_values=f["pixel_values"],
                                  max_new_tokens=7, do_sample=True, temperature=1.0, return_dict_in_generate=True,
                                  output_logits=True,
                                  logits_processor=LogitsProcessorList([AllowedTokensLogitsProcessor()]))
    gen = out.sequences[:, -7:]                                                       # [1,7]
    gl = torch.stack(out.logits, 1)[:, :, LO:HI]                                      # [1,7,256] (bf16)
    lp_store_bf16 = torch.gather(F.log_softmax(gl, -1), 2, (gen - LO).unsqueeze(-1)).sum()
    lp_store_fp32 = torch.gather(F.log_softmax(gl.float(), -1), 2, (gen - LO).unsqueeze(-1)).sum()

    # (a) 재토큰화 경로 (원 evaluate_actions)
    fe = policy._preprocess_obs(x, gen)
    retok = fe["input_ids"][:, -8:-1]
    with torch.no_grad():
        lp_eval, _e, _v = policy.vla.evaluate_action(**fe, unnorm_key="bridge_orig")

    # (b) 같은 토큰을 직접 붙인 한 번의 forward
    ids = torch.cat([f["input_ids"], gen, torch.tensor([[2]], device=dev)], 1)
    am = torch.ones_like(ids)
    with torch.no_grad():
        full = policy.vla(input_ids=ids, attention_mask=am, pixel_values=f["pixel_values"]).logits
    fl = full[:, -9:-2, LO:HI]                                                        # 행동 토큰을 예측한 위치
    lp_full_bf16 = torch.gather(F.log_softmax(fl, -1), 2, (gen - LO).unsqueeze(-1)).sum()
    lp_full_fp32 = torch.gather(F.log_softmax(fl.float(), -1), 2, (gen - LO).unsqueeze(-1)).sum()
    rows.append(dict(
        retokenized_equal=bool(torch.equal(retok, gen)),
        gen_tokens=gen[0].tolist(), retok_tokens=retok[0].tolist(),
        logit_maxdiff_gen_vs_full=float((gl.float() - fl.float()).abs().max()),
        logit_maxdiff_pos0=float((gl[:, 0].float() - fl[:, 0].float()).abs().max()),
        lp_store_bf16=float(lp_store_bf16), lp_store_fp32=float(lp_store_fp32),
        lp_eval_original=float(lp_eval), lp_full_bf16=float(lp_full_bf16), lp_full_fp32=float(lp_full_fp32)))
    r = rows[-1]
    print(f"[{i}] retok_equal={r['retokenized_equal']} logit maxdiff gen-vs-full {r['logit_maxdiff_gen_vs_full']:.3f} "
          f"(pos0 {r['logit_maxdiff_pos0']:.3f}) | store bf16 {r['lp_store_bf16']:.3f} fp32 {r['lp_store_fp32']:.3f} | "
          f"eval {r['lp_eval_original']:.3f} | full bf16 {r['lp_full_bf16']:.3f} fp32 {r['lp_full_fp32']:.3f}", flush=True)
w.env.close()
summary = dict(
    n=len(rows), retokenization_mismatches=sum(not r["retokenized_equal"] for r in rows),
    max_store_minus_eval=max(abs(r["lp_store_bf16"] - r["lp_eval_original"]) for r in rows),
    max_eval_minus_full_bf16=max(abs(r["lp_eval_original"] - r["lp_full_bf16"]) for r in rows),
    max_store_minus_full_bf16=max(abs(r["lp_store_bf16"] - r["lp_full_bf16"]) for r in rows),
    max_store_minus_full_fp32softmax=max(abs(r["lp_store_fp32"] - r["lp_full_fp32"]) for r in rows),
    max_bf16_vs_fp32_softmax_store=max(abs(r["lp_store_bf16"] - r["lp_store_fp32"]) for r in rows),
    max_logit_diff_gen_vs_full=max(r["logit_maxdiff_gen_vs_full"] for r in rows))
print("SUMMARY", json.dumps(summary), flush=True)
out_json.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2), encoding="utf-8")
