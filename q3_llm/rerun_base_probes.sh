#!/bin/bash
# Waits for run_eval.sh, then reruns the probes of the base checkpoint (its first probe run failed on a gated dataset).
cd ~/cs521_hw4/q3_llm
export HF_HOME=~/cs521_hw4/hf_cache
until grep -q "ALL DONE" logs_main.txt; do sleep 60; done
source <(sed -n '/^free_gpu()/,/^}/p' run_eval.sh)
gpu=$(free_gpu); while [ -z "$gpu" ]; do sleep 120; gpu=$(free_gpu); done
CUDA_VISIBLE_DEVICES=$gpu ~/cs521_hw4/.venv/bin/python probes.py base allenai/OLMo-2-0425-1B 0 results/probes_base.json > logs/base.probes.log 2>&1
rm -rf $HF_HOME/hub/models--allenai--OLMo-2-0425-1B
echo "BASE PROBES DONE"
