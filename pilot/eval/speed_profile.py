"""정책 추론 한 스텝의 시간 분해 (원 OpenVLAPolicy.get_action 경로).

전처리(processor) / generate / 그 밖을 나눠 재고, generate는 torch.profiler로 CPU·GPU 시간을 본다.
사용: python speed_profile.py <ckpt_dir> [unnorm_key]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_eval  # noqa: E402  (win_compat, 환경 변수 포함)
import torch  # noqa: E402

ckpt = sys.argv[1]
unnorm = sys.argv[2] if len(sys.argv) > 2 else "bridge_orig"
policy, unnorm_state = p2_eval.load_policy(ckpt, unnorm, 0.6)
w = p2_eval.WinSimplerWrapper("PutOnPlateInScene25Carrot-v1", unnorm_state, seed=0)
obs_img, instr, _ = w.reset_with_episode("train", p2_eval.episode_id_for(0, 0))
dev = torch.device("cuda:0")
x = dict(image=obs_img.to(dev), task_description=instr)


def timed(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


with torch.no_grad():
    feats = policy._preprocess_obs(x)
    print("input_ids", tuple(feats["input_ids"].shape), "pixel_values", tuple(feats["pixel_values"].shape),
          feats["pixel_values"].dtype)
    t_pre = timed(lambda: policy._preprocess_obs(x))
    t_gen = timed(lambda: policy.vla.predict_action_batch(**feats, unnorm_key=unnorm, do_sample=True, temperature=0.6))
    t_all = timed(lambda: policy.get_action(x, deterministic=True))
    base = policy.vla.base_model.model if hasattr(policy.vla, "base_model") else policy.vla
    t_gen_nohs = timed(lambda: base.generate(
        feats["input_ids"], attention_mask=feats["attention_mask"], pixel_values=feats["pixel_values"],
        max_new_tokens=7, do_sample=True, temperature=0.6))
print(f"preprocess {t_pre:.1f} ms | predict_action_batch {t_gen:.1f} ms | get_action total {t_all:.1f} ms | "
      f"plain generate (no PEFT wrapper, no hidden states) {t_gen_nohs:.1f} ms")

from torch.profiler import ProfilerActivity, profile  # noqa: E402

with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        policy.get_action(x, deterministic=True)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
ev = prof.key_averages()
cuda_total = sum(e.self_cuda_time_total for e in ev) / 3 / 1000
print(f"GPU kernel time per step ~{cuda_total:.1f} ms")
