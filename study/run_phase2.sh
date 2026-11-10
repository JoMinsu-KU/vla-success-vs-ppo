#!/usr/bin/env bash
# 본실험 2단계 실행 (H200 1개, 1단계 G-mid 뒤). 고정 복제본에서 실행할 것 (README 2절).
#   bash study/run_phase2.sh --smoke   # 먼저: 축소 시험 + 동등성 검사 (~/vla_main/smoke, 1단계 스모크 결과 필요)
#   bash study/run_phase2.sh           # 본 실행 (태그 gpre + public/gmid.json 필요). 끊기면 같은 명령으로 이어 간다
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
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"
python "$REPO/study/preflight.py" $SMOKE || { echo "preflight 실패 - 위 FAIL 항목 확인"; exit 1; }
python "$REPO/study/runner.py" --phase 2 $SMOKE
