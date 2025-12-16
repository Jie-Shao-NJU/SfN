#!/bin/bash
# Example script for Delta Stethoscope
# Analyzes weight differences between a base model and its reasoning variant

BASE_MODEL="Qwen/Qwen2.5-Math-1.5B"
REASONING_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

python compare/eval.py \
  --base_model ${BASE_MODEL} \
  --reasoning_model ${REASONING_MODEL} \
  --output_dir ./delta_results

echo "Weight difference analysis complete. Results saved to ./delta_results"
