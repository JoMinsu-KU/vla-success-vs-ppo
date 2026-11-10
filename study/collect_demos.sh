#!/usr/bin/env bash
# 모션 플래너 시연 생성 (RL4VLA collect_simpler, 원 명령과 같은 인자). A1: 16400 100, W2: 2048 200.
#   bash study/collect_demos.sh <num_traj> <seed> <record_dir> [procs=4]
# 컨테이너 제약(메모리 26 GiB, CPU 4코어): 우선순위 nice 19, cgroup anon > 25 GB면 일시정지·21 GB 아래면 재개.
set -u
N=$1; SEED=$2; OUT=$3; P=${4:-4}
source ${VLA_G2_DIR:-$HOME/vla_g2}/g2env.sh
cd ${VLA_G2_DIR:-$HOME/vla_g2}/RL4VLA/ManiSkill
T0=$(date +%s)
nice -n 19 python -m mani_skill.examples.motionplanning.widowx.collect_simpler -e "PutOnPlateInScene25Main-v3" \
  --save_data --num_procs $P --num_traj $N --seed=$SEED --record-dir $OUT &
MAIN=$!
paused=0
while kill -0 $MAIN 2>/dev/null; do
  anon=$(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat)
  kids="$MAIN $(pgrep -P $MAIN) $(for c in $(pgrep -P $MAIN); do pgrep -P $c; done)"
  if [ $paused -eq 0 ] && [ $anon -gt 25000000000 ]; then kill -STOP $kids; paused=1; echo "$(date +%T) PAUSE anon=$anon"; fi
  if [ $paused -eq 1 ] && [ $anon -lt 21000000000 ]; then kill -CONT $kids; paused=0; echo "$(date +%T) RESUME anon=$anon"; fi
  sleep 2
done
wait $MAIN; rc=$?
D=$OUT/PutOnPlateInScene25Main-v3/$N/data
echo "COLLECT DONE rc=$rc $(( $(date +%s)-T0 ))s files=$(ls $D | wc -l)"
