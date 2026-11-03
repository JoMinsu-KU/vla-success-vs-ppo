#!/usr/bin/env bash
# G2 S3: 공개 체크포인트를 원 RL4VLA 평가 명령(README "Evaluate", train_ms3_ppo.py --only_render) 그대로 돌린다.
# 원 코드는 고치지 않는다. 실행마다 WANDB_DIR를 따로 두어 결과 폴더를 (체크포인트, env_id, 시드)에 대응시킨다.
# 사용 (SimplerEnv 폴더에서): bash g2_s3_eval_original.sh <체크포인트 폴더> <출력 폴더> [phase]
#   phase: rlsft (RL·SFT × 13 env_id(=16조건) × 시드 0·1·2) | warmup_core (웜업 × Carrot·실행 3) | warmup_rest | all
# 이미 끝난 실행(stats.yaml 있음)은 건너뛴다. 실행 목록·시간은 <출력 폴더>/runs.tsv
set -uo pipefail
CK=${1:?ckpt dir}; OUT=${2:?out dir}; PHASE=${3:-all}
mkdir -p "$OUT"
[ -f "$HOME/vla_g2/g2env.sh" ] && source "$HOME/vla_g2/g2env.sh"  # conda rlvla_env + 그래픽 라이브러리 + 스레드 수

ALL_ENVS=(PutOnPlateInScene25VisionImage-v1 PutOnPlateInScene25VisionTexture03-v1 PutOnPlateInScene25VisionTexture05-v1
  PutOnPlateInScene25VisionWhole03-v1 PutOnPlateInScene25VisionWhole05-v1
  PutOnPlateInScene25Carrot-v1 PutOnPlateInScene25Plate-v1 PutOnPlateInScene25Instruct-v1
  PutOnPlateInScene25MultiCarrot-v1 PutOnPlateInScene25MultiPlate-v1
  PutOnPlateInScene25Position-v1 PutOnPlateInScene25EEPose-v1 PutOnPlateInScene25PositionChangeTo-v1)
CORE_ENVS=(PutOnPlateInScene25Carrot-v1 PutOnPlateInScene25Position-v1 PutOnPlateInScene25EEPose-v1 PutOnPlateInScene25PositionChangeTo-v1)
REST_ENVS=(PutOnPlateInScene25VisionImage-v1 PutOnPlateInScene25VisionTexture03-v1 PutOnPlateInScene25VisionTexture05-v1
  PutOnPlateInScene25VisionWhole03-v1 PutOnPlateInScene25VisionWhole05-v1
  PutOnPlateInScene25Plate-v1 PutOnPlateInScene25Instruct-v1 PutOnPlateInScene25MultiCarrot-v1 PutOnPlateInScene25MultiPlate-v1)

run() {  # name ckpt_path unnorm env seed
  local name=$1 ck=$2 un=$3 env=$4 seed=$5
  local d="$OUT/runs/$name/$env/s$seed"
  if ls "$d"/wandb/offline-run-*/glob/vis_0_test/stats.yaml >/dev/null 2>&1; then echo "[skip] $name $env s$seed"; return; fi
  # GPU시간 상한 (G2 명세 S3: 10시간). 웜업 실행은 누적 시간 + 최근 평균 1회가 상한을 넘으면 건너뛴다 (웜업부터 뺀다)
  if [[ $name == warmup ]]; then
    local used=$(awk -F'\t' '{s+=$7} END{print s+0}' "$OUT/runs.tsv" 2>/dev/null)
    local avg=$(tail -n 6 "$OUT/runs.tsv" | awk -F'\t' '{s+=$7;n++} END{print (n?s/n:600)}')
    if awk -v u="$used" -v a="$avg" 'BEGIN{exit !(u+a > 36000)}'; then echo "[budget-skip] $name $env s$seed used=${used}s"; return; fi
  fi
  rm -rf "$d"; mkdir -p "$d"
  local t0=$(date +%s)
  WANDB_DIR="$d" CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python simpler_env/train_ms3_ppo.py \
    --vla_path="$ck" --vla_unnorm_key="$un" --vla_load_path="" \
    --env_id="$env" --seed="$seed" --buffer_inferbatch=64 --no_wandb --only_render > "$d/run.log" 2>&1
  local rc=$? t1=$(date +%s)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$env" "$seed" "$rc" "$t0" "$t1" "$((t1-t0))" >> "$OUT/runs.tsv"
  echo "[done] $name $env s$seed rc=$rc $((t1-t0))s"
}

RL="$CK/openvla-7b-rlvla-rl"; SFT="$CK/openvla-7b-rlvla-sft_16k"; WU="$CK/openvla-7b-rlvla-warmup"
if [[ $PHASE == rlsft || $PHASE == all ]]; then
  for seed in 0 1 2; do for env in "${ALL_ENVS[@]}"; do
    run rl "$RL" bridge_orig "$env" "$seed"; run sft "$SFT" sft "$env" "$seed"
  done; done
fi
if [[ $PHASE == warmup_core || $PHASE == all ]]; then
  for seed in 0 1 2; do for env in "${CORE_ENVS[@]}"; do run warmup "$WU" bridge_orig "$env" "$seed"; done; done
fi
if [[ $PHASE == warmup_rest || $PHASE == all ]]; then
  for seed in 0 1 2; do for env in "${REST_ENVS[@]}"; do run warmup "$WU" bridge_orig "$env" "$seed"; done; done
fi
echo "S3 phase $PHASE done"
