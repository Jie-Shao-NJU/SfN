#!/bin/bash
# ==== 配置基本参数 ====
lr=1e-5
epochs=5
batch_size=16
weight_decay=1e-4
train_dataset_name="s1K_tokenized"
uid="$(date +%Y%m%d_%H%M%S)"

# ==== 解析命令行参数 ====
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --weight_decay) weight_decay="$2"; shift 2 ;;
        --train_dataset_name) train_dataset_name="$2"; shift 2 ;;
        --uid) uid="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# ==== 获取节点信息 ====
node_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nnodes=${#node_array[@]}
head_node=${node_array[0]}
head_node_ip=$(ssh $head_node hostname --ip-address)
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600

# ==== 多机环境变量设置 ====
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500
export WORLD_SIZE=$(($SLURM_NNODES * $(nvidia-smi -L | wc -l)))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# ==== 计算 grad accumulation steps ====
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=$((batch_size/(gpu_count * nnodes)))

echo "Number of nodes: $nnodes"
echo "Number of GPUs per node: $gpu_count"
echo "Head node IP: $head_node_ip"

run_name="qwen_${train_dataset_name}_bs${batch_size}_lr${lr}_epoch${epochs}_wd${weight_decay}_${uid}"

# ==== 启动训练 ====
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3-2.yaml \
    train/sft2.py \
    --block_size=32768 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/${train_dataset_name}" \
    --model_name="ckpt/Qwen2.5-14B-Instruct" \
    --warmup_ratio=0.05 \
    --report_to="none" \
    --fsdp_config="train/fsdp_config_qwen_cpu.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/${run_name}" \
    --push_to_hub=false \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
