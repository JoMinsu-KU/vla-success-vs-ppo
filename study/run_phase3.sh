#!/usr/bin/env bash
# 본실험 3단계 실행 (E2-b 분기, E1 핵심 갈래가 모두 끝난 뒤). 고정 복제본에서 실행할 것 (README 2절).
#   bash study/run_phase3.sh --smoke   # 먼저: 축소 분기 + C± 비트 일치 검사 (~/vla_main/smoke)
#   bash study/run_phase3.sh [--a3plus] # 본 실행. --a3plus는 E2-b 판정(|D_C| ≥ 5%p) 뒤에만
# 환경 변수 (기본값은 G2 인스턴스 배치):
#   VLA_G2_DIR   ~/vla_g2                      (g2env.sh, RL4VLA, checkpoints가 있는 곳)
#   VLA_MAIN_WORK ~/vla_main                   (본실험 작업 폴더: 체크포인트·기록)
#   VLA_W1_CKPT  $VLA_G2_DIR/checkpoints/openvla-7b-rlvla-warmup
set -uo pipefail
G2=${VLA_G2_DIR:-$HOME/vla_g2}
export VLA_MAIN_WORK=${VLA_MAIN_WORK:-$HOME/vla_main}
export VLA_W1_CKPT=${VLA_W1_CKPT:-$G2/checkpoints/openvla-7b-rlvla-warmup}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$G2/g2env.sh"
cd "$G2/RL4VLA/SimplerEnv"
mkdir -p "$VLA_MAIN_WORK"
SMOKE=""; A3PLUS=""; for x in "$@"; do [ "$x" = "--smoke" ] && SMOKE="--smoke"; [ "$x" = "--a3plus" ] && A3PLUS="--a3plus"; done
python "$REPO/study/preflight.py" $SMOKE || { echo "preflight 실패 - 위 FAIL 항목 확인"; exit 1; }
python "$REPO/study/runner.py" --phase 3 $SMOKE ${A3PLUS:-}
