#!/usr/bin/env bash
# G2 S3 우선순위 실행 (이탈 D-G2-1): RL·SFT 16조건 × 시드 3이 10 GPU시간 상한을 넘으므로(실측 외삽 약 12.7시간)
# 판정에 쓰는 조건부터 시드를 채운다. 모든 실행에 상한 가드(누적 + 최근 평균 1회 > 10시간이면 건너뜀).
# 순서: ① 시드 0 RL·SFT 16조건 ② 시드 1·2 RL·SFT 판정 조건(Carrot train/test, Position, EEPose, PositionChangeTo)
#       ③ 웜업 시드 0 판정 조건 ④ 시드 1·2 RL·SFT 나머지 조건 ⑤ 웜업 시드 1·2 판정 조건
# 사용 (SimplerEnv 폴더에서): bash g2_s3_priority.sh <체크포인트 폴더> <출력 폴더>
set -uo pipefail
CK=${1:?ckpt dir}; OUT=${2:?out dir}
source "$HOME/vla_g2/g2env.sh"
CAP=36000
ALL=(PutOnPlateInScene25VisionImage-v1 PutOnPlateInScene25VisionTexture03-v1 PutOnPlateInScene25VisionTexture05-v1
  PutOnPlateInScene25VisionWhole03-v1 PutOnPlateInScene25VisionWhole05-v1
  PutOnPlateInScene25Carrot-v1 PutOnPlateInScene25Plate-v1 PutOnPlateInScene25Instruct-v1
  PutOnPlateInScene25MultiCarrot-v1 PutOnPlateInScene25MultiPlate-v1
  PutOnPlateInScene25Position-v1 PutOnPlateInScene25EEPose-v1 PutOnPlateInScene25PositionChangeTo-v1)
CORE=(PutOnPlateInScene25Carrot-v1 PutOnPlateInScene25Position-v1 PutOnPlateInScene25EEPose-v1 PutOnPlateInScene25PositionChangeTo-v1)
REST=(PutOnPlateInScene25VisionImage-v1 PutOnPlateInScene25VisionTexture03-v1 PutOnPlateInScene25VisionTexture05-v1
  PutOnPlateInScene25VisionWhole03-v1 PutOnPlateInScene25VisionWhole05-v1
  PutOnPlateInScene25Plate-v1 PutOnPlateInScene25Instruct-v1 PutOnPlateInScene25MultiCarrot-v1 PutOnPlateInScene25MultiPlate-v1)
RL="$CK/openvla-7b-rlvla-rl"; SFT="$CK/openvla-7b-rlvla-sft_16k"; WU="$CK/openvla-7b-rlvla-warmup"

run() {  # name ckpt unnorm env seed
  local name=$1 ck=$2 un=$3 env=$4 seed=$5
  local d="$OUT/runs/$name/$env/s$seed"
  if ls "$d"/wandb/offline-run-*/glob/vis_0_test/stats.yaml >/dev/null 2>&1; then return; fi
  local used=$(awk -F'\t' '{s+=$7} END{print s+0}' "$OUT/runs.tsv")
  local avg=$(tail -n 6 "$OUT/runs.tsv" | awk -F'\t' '{s+=$7;n++} END{print (n?s/n:600)}')
  if awk -v u="$used" -v a="$avg" 'BEGIN{exit !(u+a > '$CAP')}'; then echo "[budget-skip] $name $env s$seed used=${used}s"; return; fi
  rm -rf "$d"; mkdir -p "$d"
  local t0=$(date +%s)
  WANDB_DIR="$d" CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python simpler_env/train_ms3_ppo.py --vla_path="$ck" --vla_unnorm_key="$un" --vla_load_path="" \
    --env_id="$env" --seed="$seed" --buffer_inferbatch=64 --no_wandb --only_render > "$d/run.log" 2>&1
  local rc=$? t1=$(date +%s)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$env" "$seed" "$rc" "$t0" "$t1" "$((t1-t0))" >> "$OUT/runs.tsv"
  echo "[done] $name $env s$seed rc=$rc $((t1-t0))s"
}
for env in "${ALL[@]}"; do run rl "$RL" bridge_orig "$env" 0; run sft "$SFT" sft "$env" 0; done
for seed in 1 2; do for env in "${CORE[@]}"; do run rl "$RL" bridge_orig "$env" $seed; run sft "$SFT" sft "$env" $seed; done; done
for env in "${CORE[@]}"; do run warmup "$WU" bridge_orig "$env" 0; done
for seed in 1 2; do for env in "${REST[@]}"; do run rl "$RL" bridge_orig "$env" $seed; run sft "$SFT" sft "$env" $seed; done; done
for seed in 1 2; do for env in "${CORE[@]}"; do run warmup "$WU" bridge_orig "$env" $seed; done; done
echo "S3 phase warmup_core done (priority order)"
