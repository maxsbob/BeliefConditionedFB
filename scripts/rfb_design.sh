#!/usr/bin/env bash
# RFB design study on 20 training layouts (one seed, 300K FB steps).
set -u
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COMMON="num_train_layouts=20 context_len=100 seed=0 fb_steps=300000 eval.interval=50000 save_checkpoint=false"
run() { gpu=$1; name=$2; shift 2
  [ -f runs/design/$name/final.json ] && return
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python scripts/train_fourrooms.py $COMMON out_dir=runs/design/$name "$@" > logs/design_$name.log 2>&1
  echo "done $name"; }
# 1) pre-train the shared encoder once
run 0 encoder_only agent=bfb fb_steps=0
# 2) variants
run 0 bfb_raw agent=bfb agent.standardize_h=false &
run 0 bfb agent=bfb &
run 0 rfb_zprior agent=rfb agent.z_sampling=vmf agent.kappa=50 &
run 0 rfb_bk300 agent=rfb agent.z_sampling=uniform agent.belief_kappa=300 &
run 1 rfb_bk100 agent=rfb agent.z_sampling=uniform agent.belief_kappa=100 &
run 1 rfb_bk30 agent=rfb agent.z_sampling=uniform agent.belief_kappa=30 &
run 1 rfb_bk10 agent=rfb agent.z_sampling=uniform agent.belief_kappa=10 &
run 1 rfb_bk100_zprior agent=rfb agent.z_sampling=vmf agent.kappa=50 agent.belief_kappa=100 &
wait
echo ALL DONE
