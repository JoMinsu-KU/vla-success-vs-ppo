#!/usr/bin/env bash
# P1-b: rlvla_env에 ManiSkill을 넣는다. torch·numpy가 바뀌지 않도록 제약 파일을 쓴다.
# (P1 본 설치에서 RL4VLA README의 ManiSkill로 교체할 수 있다.)
set -eu
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rlvla_env
printf 'torch==2.2.0\ntorchvision==0.17.0\nnumpy<2\n' > /tmp/p1b_constraints.txt
pip install --no-input mani_skill -c /tmp/p1b_constraints.txt > /tmp/p1b_ms_install.log 2>&1 || { tail -20 /tmp/p1b_ms_install.log; exit 1; }
tail -1 /tmp/p1b_ms_install.log
python -c "import torch, numpy, mani_skill, sapien; print('torch', torch.__version__, 'numpy', numpy.__version__, 'mani_skill', mani_skill.__version__, 'sapien', sapien.__version__)"
cat ~/miniforge3/envs/p1b_mesa_probe/share/vulkan/icd.d/lvp_icd.x86_64.json
