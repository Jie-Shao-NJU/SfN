#!/bin/bash
# environ: llamagen
set -x

CONFIG=${1}


GPUS=${2:-0}
GPUS_PER_NODE=${3:-0}
# PARTITION=${4:-"VC2"}
PARTITION=${4:-"INTERN2"}
QUOTA_TYPE=${5:-"auto"}
JOB_NAME=${6:-"vl_sj"}

CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

# SRUN_ARGS=${SRUN_ARGS:-" --jobid=4947146"} # 3816987

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=${MASTER_PORT:-32436}
export LAUNCHER=slurm
# export NCCL_DEBUG=INFO
# export TF_CPP_MIN_LOG_LEVEL=3
# unset CUDA_LAUNCH_BLOCKING
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="+dynamo" 
# export TORCHDYNAMO_VERBOSE=1
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export BATCH_SIZE

export LOGGING_GRAD_NORM
export DEBUG
export SKIP_LOG_IMAGE
export DEEPSPEED_TIMEOUT=60
export TEXT_NUM_IMAGE

# enable for sampling only
# export LD_LIBRARY_PATH="/mnt/petrelfs/share_data/tianchangyao.p/cuda/cuda-11.7/lib64":$LD_LIBRARY_PATH

SUFFIX=$(date '+%Y%m%d%H%M')
SRUN_ARGS=${SRUN_ARGS:-"-w SH-IDC1-10-140-37-74"} # 157 74 14 // 5 11

# -w HOST-10-140-66-[138-142,144-145,147-148,150,174,177-178,194] \
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model="ckpts/qwen_s1K_tokenized_bs16_lr1e-5_epoch5_wd1e-4_20250512_194335"

proxy_on_usa

srun -p ${PARTITION} \
  --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  lm_eval --model vllm --model_args pretrained=${model},tokenizer=ckpt/Qwen2.5-14B-Instruct,dtype=float32,tensor_parallel_size=8 --tasks aime24_figures,aime24_nofigures,openai_math,gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path forcingignore2wait_attn --log_samples --gen_kwargs "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=2,thinking_n_ignore_str=Wait"


  # lm_eval --model vllm --model_args pretrained=ckpt/Qwen2.5-14B-Instruct,tokenizer=ckpt/Qwen2.5-14B-Instruct,dtype=float32,tensor_parallel_size=8 --tasks aime24_figures,aime24_nofigures,openai_math,gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path qwen14b --log_samples --gen_kwargs "max_gen_toks=32768"



  # lm_eval --model vllm --model_args pretrained=ckpt/Qwen2.5-32B-Instruct,tokenizer=ckpt/Qwen2.5-32B-Instruct,dtype=float32,tensor_parallel_size=8 --tasks aime24_figures,aime24_nofigures,openai_math,gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path qwen --log_samples --gen_kwargs "max_gen_toks=32768"


  
#   bash train/sft.sh

# lm_eval --model vllm --model_args pretrained="Qwen/Qwen2.5-32B-Instruct",tokenizer="Qwen/Qwen2.5-32B-Instruct",dtype=float32,tensor_parallel_size=8 --tasks gpqa_diamond_openai --batch_size auto --apply_chat_template --output_path qwen --log_samples --gen_kwargs "max_gen_toks=32768"