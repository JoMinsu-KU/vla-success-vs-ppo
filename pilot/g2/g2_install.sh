#!/usr/bin/env bash
# G2 S1: rlvla_env 설치 (RL4VLA README의 Linux 절차 + 파일럿에서 고정한 버전 제약).
# 사용: bash g2_install.sh <작업 폴더>   예: bash g2_install.sh ~/vla_g2   (로그: <작업 폴더>/install.log)
# 시스템 설정(드라이버, apt 패키지, sudo)은 건드리지 않는다.
set -euo pipefail
WORK=${1:?work dir}
mkdir -p "$WORK"; cd "$WORK"
exec > >(tee -a install.log) 2>&1
eval "$(conda shell.bash hook)"

# 1. 코드: 연구 저장소와 RL4VLA (커밋 de46e01 고정)
[ -d vla-success-vs-ppo ] || git clone https://github.com/JoMinsu-KU/vla-success-vs-ppo.git
[ -d RL4VLA ] || git clone https://github.com/gen-robot/RL4VLA.git
git -C RL4VLA checkout de46e01
echo "research repo: $(git -C vla-success-vs-ppo rev-parse HEAD)  RL4VLA: $(git -C RL4VLA rev-parse HEAD)"

# 2. conda 환경 (README와 같은 python 3.10)
conda env list | grep -q "^rlvla_env " || conda create -n rlvla_env -y python=3.10
conda activate rlvla_env
C="$WORK/vla-success-vs-ppo/pilot/scripts/constraints_torch220.txt"

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install "setuptools<70" wheel
(cd RL4VLA/openvla && pip install -e . -c "$C")
pip install -U tyro -c "$C"
pip install datasets==3.3.2 -c "$C"
# flash-attn: README가 지정한 사전 빌드 휠 (torch 2.2, cu12, python 3.10)
FA=flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
python -c "import flash_attn" 2>/dev/null || { wget -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$FA" && pip install "$FA" && rm -f "$FA"; }
(cd RL4VLA/ManiSkill && pip install -e . -c "$C" --no-build-isolation)
(cd RL4VLA/SimplerEnv && pip install -e . -c "$C" --no-build-isolation)
# 파일럿과 같은 OpenVLA 계열 버전 고정 (transformers 4.40.1, peft 0.11.1, timm 0.9.10 등)
pip install -c "$WORK/vla-success-vs-ppo/pilot/scripts/constraints_openvla.txt" -c "$C" \
  transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 peft==0.11.1 sentencepiece==0.1.99 scipy pandas

# 3. ManiSkill 자산 (bridge 장면, WidowX)
python -m mani_skill.utils.download_asset bridge_v2_real2sim -y
python -m mani_skill.utils.download_asset widowx250s -y

# 4. 확인과 기록
python - <<'EOF'
import torch, numpy, transformers, peft, timm, sapien, mani_skill, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "n_gpu", torch.cuda.device_count())
print("numpy", numpy.__version__, "transformers", transformers.__version__, "peft", peft.__version__, "timm", timm.__version__)
print("sapien", sapien.__version__, "mani_skill", mani_skill.__version__, "flash_attn", flash_attn.__version__)
EOF
pip freeze > "$WORK/rlvla_env_pip_freeze.txt"
conda list --explicit > "$WORK/rlvla_env_conda_explicit.txt"
echo "S1 done"
