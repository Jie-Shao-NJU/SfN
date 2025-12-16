#!/bin/bash
# Example script for Merge Stethoscope
# Merges o_proj from reasoning model into base model

BASE_MODEL="Qwen/Qwen2.5-Math-1.5B"
REASONING_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR="./merged_model"

python compare/merge.py \
  --base ${BASE_MODEL} \
  --reasoning ${REASONING_MODEL} \
  --output ${OUTPUT_DIR}

echo "Model merging complete. Merged model saved to ${OUTPUT_DIR}"
