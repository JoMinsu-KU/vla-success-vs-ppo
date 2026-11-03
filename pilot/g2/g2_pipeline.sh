#!/usr/bin/env bash
# G2 S4~S9 순차 실행기 (S3 종료를 기다린 뒤). 단계별 시작·종료 시각을 <작업 폴더>/ledger.tsv에 기록한다 (GPU시간 장부).
# 사용: bash g2_pipeline.sh   (작업 폴더 ~/vla_g2, 체크포인트 ~/vla_g2/checkpoints)
set -uo pipefail
source "$HOME/vla_g2/g2env.sh"
W=$HOME/vla_g2; R=$W/vla-success-vs-ppo/pilot/g2; CK=$W/checkpoints; W1=$CK/openvla-7b-rlvla-warmup
cd "$W/RL4VLA/SimplerEnv"
stage() {  # 이름 명령...
  local name=$1; shift
  if grep -q "^$name	.*	0$" "$W/ledger.tsv" 2>/dev/null; then echo "[skip] $name"; return 0; fi
  local t0=$(date +%s); echo "[start] $name $(date -Is)"
  "$@" > "$W/logs/$name.log" 2>&1; local rc=$?
  local t1=$(date +%s); printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$t0" "$t1" "$((t1-t0))" "$rc" >> "$W/ledger.tsv"
  echo "[end] $name rc=$rc $((t1-t0))s"; return $rc
}
mkdir -p "$W/logs"
until grep -q "S3 phase warmup_core done" "$W/s3/driver.log" 2>/dev/null; do sleep 60; done
stage S4 python "$R/g2_s4_eval_controlled.py" "$CK" "$W/s4"
stage S5 python "$R/g2_s5_scale.py" "$W1" "$W/s5"
stage S6_flash_det python "$R/g2_s6_determinism.py" "$W1" "$W/s6" --path flash_det
stage S6_math python "$R/g2_s6_determinism.py" "$W1" "$W/s6" --path math
# 사전 고정 규칙(docs/G2_prereg.md): 두 시작점 A/A 모두 비트 동일 + 엄격 오류 없음인 경로 중 업데이트 시간이 짧은 쪽
P=$(python - <<'EOF'
import json, pathlib
best = None
for p in ("flash_det", "math"):
    f = pathlib.Path.home() / "vla_g2/s6" / f"g2_s6_{p}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    ok = d.get("strict_error") is None and all(d.get(k, {}).get("lora_bitwise_equal") and d.get(k, {}).get("value_bitwise_equal")
                                               for k in ("aa_fresh", "aa_after_one_round"))
    if ok:
        t = sum(d["aa_fresh"]["update_sec"] + d["aa_after_one_round"]["update_sec"]) / 4
        if best is None or t < best[1]:
            best = (p, t)
print(best[0] if best else "NONE")
EOF
)
echo "S9 path = $P" | tee "$W/s9_path.txt"
[ "$P" = "NONE" ] && { echo "no deterministic path passed; stopping before S9"; exit 1; }
stage S7_W1 python "$R/g2_s7_c10.py" "$W1" "$W/s7" --label W1 --path "$P"
stage S9_A3 python "$R/g2_s9_train.py" "$W1" "$W/s9" --arm A3 --rounds 50 --path "$P"
stage S7_A3r10 python "$R/g2_s7_c10.py" "$W1" "$W/s7" --label A3_r10 --lora "$W/s9/ckpt/A3_r10.pt" --path "$P" --blind-Y
stage S9_A2p python "$R/g2_s9_train.py" "$W1" "$W/s9" --arm A2p --rounds 50 --path "$P"
echo "PIPELINE DONE $(date -Is)"
