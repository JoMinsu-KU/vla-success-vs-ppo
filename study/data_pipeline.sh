#!/usr/bin/env bash
# 본실험 데이터 준비 (CPU, nice 19, 끝난 단계는 건너뜀): A1 시연 16,400 → W2 시연 2,048 → A1 전처리 → W2 사다리 전처리
#   bash study/data_pipeline.sh            (로그: ~/vla_main_data/data_pipeline.log)
# 설계: study/A1_design.md 1절, study/W2_design.md 1절
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=${VLA_MAIN_DATA:-$HOME/vla_main_data}
W1=${VLA_W1_CKPT:-$HOME/vla_g2/checkpoints/openvla-7b-rlvla-warmup}
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*"; }
count() { ls $1 2>/dev/null | wc -l; }
A1D=$D/mp_collect/PutOnPlateInScene25Main-v3/16400/data
W2D=$D/mp_collect_w2/PutOnPlateInScene25Main-v3/2048/data
log "pipeline start, code $(git -C $REPO rev-parse --short HEAD)"
# 1. A1 시연 (이미 다른 프로세스가 생성 중이면 끝날 때까지 기다린다)
while pgrep -f "collect_simpler.*--num_traj 16400" >/dev/null; do sleep 60; done
if [ $(count $A1D) -lt 16400 ]; then log "A1 collect start"; bash $REPO/study/collect_demos.sh 16400 100 $D/mp_collect 4; fi
[ $(count $A1D) -ge 16400 ] || { log "A1 collect incomplete: $(count $A1D)"; exit 1; }
log "A1 demos: $(count $A1D)"
# 2. W2 시연
if [ $(count $W2D) -lt 2048 ]; then log "W2 collect start"; bash $REPO/study/collect_demos.sh 2048 200 $D/mp_collect_w2 4; fi
[ $(count $W2D) -ge 2048 ] || { log "W2 collect incomplete: $(count $W2D)"; exit 1; }
log "W2 demos: $(count $W2D)"
source ${VLA_G2_DIR:-$HOME/vla_g2}/g2env.sh
# 3. A1 전처리
if [ ! -f $D/a1/train/meta.json ] || [ ! -f $D/a1/val/meta.json ]; then
  log "A1 prepare start"; nice -n 19 python $REPO/study/a1_prepare.py $A1D $D/a1 $W1 --procs 2 || { log "A1 prepare failed"; exit 1; }; fi
log "A1 prepared: $(python -c "import json;m=json.load(open('$D/a1/train/meta.json'));print(m['frames'], m['sha256'])")"
# 4. W2 사다리 전처리
if [ ! -f $D/w2/prep/n2048/meta.json ]; then
  log "W2 prepare start"; nice -n 19 python $REPO/study/a1_prepare.py $W2D $D/w2/prep $W1 --procs 2 --subset 256,512,1024,2048 || { log "W2 prepare failed"; exit 1; }; fi
log "PIPELINE DONE"
