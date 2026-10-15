# RL 체크포인트 샤드 4개가 다 받아지면(다운로드 로그 기준) 실제 모델로 2 에피소드 시험을 돌린다.
$log = "D:\VLA_ReinforceLearning\checkpoints\download.log"
$deadline = (Get-Date).AddHours(4)
while ((Get-Date) -lt $deadline) {
    $n = 0
    if (Test-Path $log) { $n = (Select-String -Path $log -Pattern 'openvla-7b-rlvla-rl/model-0000\d-of-00004\.safetensors' | Where-Object { $_.Line -match 'downloaded|exists' }).Count }
    if ($n -ge 4) { break }
    Start-Sleep -Seconds 60
}
if ($n -lt 4) { "TIMEOUT waiting for RL checkpoint (shards done: $n)"; exit 1 }
"RL checkpoint ready at $(Get-Date -Format HH:mm:ss)"
$py = "C:\Users\choiLee\anaconda3\envs\rlvla_win\python.exe"
& $py -u D:\VLA_ReinforceLearning\pilot\eval\p2_eval.py --ckpt D:\VLA_ReinforceLearning\checkpoints\openvla-7b-rlvla-rl --unnorm-key bridge_orig --seeds 0 --tasks "PutOnPlateInScene25Carrot-v1:train" --n-episodes 2 --out D:\VLA_ReinforceLearning\pilot\artifacts\p2_model_smoke\rl_smoke.jsonl --video-dir D:\VLA_ReinforceLearning\pilot\artifacts\p2_model_smoke\videos --n-videos 2 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -match '^\[|Error|error|Traceback|File "' }
