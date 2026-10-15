# P2 공개 체크포인트 재현 평가 실행 (연구자가 직접 실행)
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File D:\VLA_ReinforceLearning\pilot\eval\run_p2.ps1
#
# - 순서: RL -> SFT-16k -> 웜업. 체크포인트마다 핵심 4과제 x 시드 0,1,2 x 64 에피소드.
# - 결과: pilot\artifacts\p2\p2_results.jsonl (에피소드마다 한 줄). 로그: pilot\artifacts\p2\logs\
# - 중간에 멈추거나 창을 닫아도, 같은 명령을 다시 실행하면 끝난 에피소드는 건너뛰고 이어서 한다.
# - 체크포인트 샤드(4개)가 아직 다 받아지지 않았으면 그 체크포인트는 건너뛰고 메시지를 남긴다.
# - 다른 GPU 작업과 동시에 돌리지 않는다 (속도·메모리).

$ErrorActionPreference = "Continue"
$root = "D:\VLA_ReinforceLearning"
$py = "C:\Users\choiLee\anaconda3\envs\rlvla_win\python.exe"
$out = "$root\pilot\artifacts\p2\p2_results.jsonl"
$logDir = "$root\pilot\artifacts\p2\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

$env:PYTHONWARNINGS = "ignore"
$env:USE_TF = "0"

$ckpts = @(
    @{ name = "openvla-7b-rlvla-rl";      unnorm = "bridge_orig" },
    @{ name = "openvla-7b-rlvla-sft_16k"; unnorm = "sft" },
    @{ name = "openvla-7b-rlvla-warmup";  unnorm = "bridge_orig" }
)

foreach ($c in $ckpts) {
    $dir = Join-Path "$root\checkpoints" $c.name
    $shards = @(Get-ChildItem $dir -Filter "model-0000?-of-00004.safetensors" -ErrorAction SilentlyContinue)
    $dlog = "$root\checkpoints\download.log"
    $doneShards = 0
    if (Test-Path $dlog) {
        $doneShards = @(Select-String -Path $dlog -Pattern ("gen-robot/" + $c.name + "/model-0000\d-of-00004\.safetensors") |
                        Where-Object { $_.Line -match "downloaded|exists" }).Count
    }
    if ($shards.Count -lt 4 -or $doneShards -lt 4) {
        Write-Host "[skip] $($c.name): checkpoint not fully downloaded yet (shards done: $doneShards/4). Run this script again later."
        continue
    }
    $log = Join-Path $logDir ("{0}_{1}.log" -f $c.name, (Get-Date -Format "yyyyMMdd_HHmmss"))
    Write-Host "[start] $($c.name)  log: $log"
    & $py -u "$root\pilot\eval\p2_eval.py" --ckpt $dir --unnorm-key $c.unnorm `
        --seeds 0 1 2 --tasks p2 --n-episodes 64 --out $out `
        --video-dir "$root\pilot\artifacts\p2\videos" --n-videos 2 2>&1 |
        ForEach-Object { "$_" } |
        Where-Object { $_ -notmatch "pinnochio|warnings\.warn|pynvml|VR Path|xyz_configs|quat_configs|robot_qpos|Deprecated member|joint = self\.model" } |
        Tee-Object -FilePath $log -Append
    Write-Host "[end] $($c.name)"
}
Write-Host "[all] finished. Summary:"
& $py "$root\pilot\eval\p2_summary.py" $out
