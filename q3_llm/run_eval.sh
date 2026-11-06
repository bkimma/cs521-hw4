#!/bin/bash
# Evaluate the four OLMo 2 1B checkpoints on the cluster, one at a time.
# Runs only on a GPU with no processes; deletes each checkpoint after its evaluation to save disk space.
cd ~/cs521_hw4/q3_llm
export HF_HOME=~/cs521_hw4/hf_cache
PY=~/cs521_hw4/.venv/bin/python
mkdir -p results logs

free_gpu() {
  for i in 0 1 2 3; do
    uuid=$(nvidia-smi -i $i --query-gpu=uuid --format=csv,noheader)
    procs=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
    mem=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$procs" -eq 0 ] && [ "$mem" -lt 500 ]; then echo $i; return; fi
  done
}

for spec in "base allenai/OLMo-2-0425-1B 0" "sft allenai/OLMo-2-0425-1B-SFT 1" \
            "dpo allenai/OLMo-2-0425-1B-DPO 1" "instruct allenai/OLMo-2-0425-1B-Instruct 1"; do
  set -- $spec; name=$1; repo=$2; chat=$3
  gpu=$(free_gpu)
  while [ -z "$gpu" ]; do echo "$(date) no free GPU, waiting"; sleep 120; gpu=$(free_gpu); done
  echo "$(date) $name on GPU $gpu"
  export CUDA_VISIBLE_DEVICES=$gpu
  chat_args=""
  [ "$chat" = "1" ] && chat_args="--apply_chat_template --fewshot_as_multiturn"
  $PY -m lm_eval --model hf --model_args pretrained=$repo,dtype=bfloat16 \
      --tasks gsm8k,ifeval,truthfulqa_mc1,truthfulqa_mc2 --limit 200 --batch_size 32 --seed 0 \
      --log_samples --output_path results/$name $chat_args > logs/$name.lmeval.log 2>&1
  $PY probes.py $name $repo $chat results/probes_$name.json > logs/$name.probes.log 2>&1
  rm -rf $HF_HOME/hub/models--${repo/\//--}
  echo "$(date) $name done"; df -h ~ | tail -1
done
echo ALL DONE
