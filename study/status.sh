#!/usr/bin/env bash
# 진행 상황 (공개 정보만): 계획 작업 수, 실행별 상태·라운드, 최근 로그. 맹검 폴더는 읽지 않는다.
#   bash study/status.sh [--smoke]
W=${VLA_MAIN_WORK:-$HOME/vla_main}; [ "${1:-}" = "--smoke" ] && W=$W/smoke
echo "== $W"
tail -n 15 "$W/public/progress.log" 2>/dev/null
echo "== runs"
for f in "$W"/public/runs/*.json; do [ -e "$f" ] && python -c "import json,sys;d=json.load(open('$f'));print(d['run_id'],d['status'],str(d['rounds_done'])+'/'+str(d['rounds']),d.get('gpu_h_total',''))"; done
echo "== disk"; df -h "$W" | tail -1
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
