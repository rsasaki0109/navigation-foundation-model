#!/usr/bin/env bash
# Fine-tune the learned paradigms on a 50/50 mix of apartment and multi-room house episodes.
# Starts from the apartment checkpoints (copied to checkpoints/house/) and writes there.
set -euo pipefail
export NAVFM_LAYOUT=mixed NAVFM_CKPT=checkpoints/house NAVFM_DATA=data/expert_mixed.npz
python -u scripts/collect.py 3000 data/expert_mixed.npz
python -u scripts/train.py vla --finetune
python -u scripts/train.py wam --finetune
python -u scripts/train.py wm --finetune
ONPOLICY_OFFSET=250000 ONPOLICY_ROUNDS=3 python -u scripts/train.py rwm-ep-onpolicy
