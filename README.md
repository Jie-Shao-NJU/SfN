# Who Reasons in the Large Language Models?

[![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue)](https://arxiv.org/abs/TODO)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

> **Official implementation of "Who Reasons in the Large Language Models?"**
> Jie Shao, Jianxin Wu
> NeurIPS 2025

## Overview

This repository contains the implementation of **Stethoscope for Networks (SfN)**, a suite of diagnostic tools designed to identify which components of LLMs give rise to specific abilities, particularly mathematical reasoning.

**Key Findings:**
- The **output projection (`o_proj`)** in the multi-head self-attention (MHSA) module is primarily responsible for reasoning in LLMs
- Other modules (`q_proj`, `k_proj`, `v_proj`, and MLP) contribute more to conversational fluency
- Training only `o_proj` (+LayerNorm) achieves competitive reasoning performance while being **3x faster** than full finetuning

<p align="center">
  <img src="assets/overview.png" alt="SfN Overview" width="800"/>
</p>

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
  - [Data Preparation](#data-preparation)
  - [Fine-tuning Models](#fine-tuning-models)
- [Stethoscope Tools](#stethoscope-tools)
  - [Delta Stethoscope](#delta-stethoscope)
  - [Merge Stethoscope](#merge-stethoscope)
  - [Freeze Stethoscope](#freeze-stethoscope)
  - [Destruction Stethoscope](#destruction-stethoscope)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)

## Features

- **Delta Stethoscope**: Analyze weight differences between base and reasoning models
- **Merge Stethoscope**: Selectively merge modules from different models
- **Freeze Stethoscope**: Efficiently fine-tune by freezing selected modules
- **Destruction Stethoscope**: Systematically disable modules to infer functionality

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stethoscope-for-networks.git
cd stethoscope-for-networks

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- deepspeed (for training)
- Additional dependencies listed in `requirements.txt`

## Quick Start

### 1. Delta Stethoscope: Analyze Weight Differences

Compare a base model and its reasoning-enhanced variant:

```python
import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt

# Load models
model_a = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
model_b = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Compute L2 norm of weight differences
for name, param in model_a.named_parameters():
    if 'o_proj' in name:
        diff = torch.norm(param - model_b.state_dict()[name])
        print(f"{name}: {diff.item()}")
```

### 2. Merge Stethoscope: Selective Module Merging

Merge `o_proj` from a reasoning model into a base model:

```python
from compare.merge import merge_models

# Merge only o_proj weights
merged_model = merge_models(
    base_model="Qwen/Qwen2.5-32B",
    reasoning_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    modules_to_merge=["o_proj", "norm", "lm_head", "embed_tokens"]
)

# Save merged model
merged_model.save_pretrained("./merged_model")
```

### 3. Freeze Stethoscope: Efficient Fine-tuning

Fine-tune only `o_proj` for reasoning tasks:

```bash
# Configure which modules to train in deepspeed config
bash scripts/train_freeze.sh
```

Example configuration:
```yaml
# Only train o_proj, norm, lm_head, and embed_tokens
trainable_modules:
  - "self_attn.o_proj"
  - "norm"
  - "lm_head"
  - "embed_tokens"
```

### 4. Destruction Stethoscope: Module Functionality Analysis

Systematically disable modules to understand their roles:

```python
from compare.destruction import zero_out_module, reinit_module

# Zero out o_proj in specific layers
model = zero_out_module(model, module_pattern="layers.[5-30].self_attn.o_proj")

# Test conversation capability
response = model.generate(prompt, max_length=512)
```

## Training

### Data Preparation

The `data/` directory contains scripts for collecting and processing training data from various mathematical and scientific reasoning datasets.

#### Collect Training Data

```bash
# Collect data from multiple sources (MATH, AIME, GPQA, etc.)
cd data
python collect_data.py --output_path ./processed_data
```

Key features of data processing:
- **Decontamination**: Remove test set contamination from training data
- **Featurization**: Extract features from problem-solution pairs
- **Tokenization**: Prepare data for model training
- **Multiple dataset support**: MATH, AIME, GPQA, medical QA, chemistry problems, etc.

#### Available Data Processing Scripts

- `collect_data.py`: Main data collection and processing pipeline
- `bulk_inference.py`: Batch inference for data generation
- `featurization.py`: Extract features from datasets
- `tokenization.py`: Tokenize datasets for training
- `decontaminate_util.py`: Remove test set contamination

### Fine-tuning Models

The `train/` directory contains training scripts for supervised fine-tuning (SFT) on reasoning tasks.

#### Basic Training

```bash
cd train

# Single-node training with standard settings
bash sft.sh

# Multi-node training with SLURM
bash sft_slurm.sh

# Training with Accelerate
bash sft_accelerate.sh
```

#### Training Script Options

The main training script (`sft.py`) supports:

```bash
python train/sft.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --train_file_path simplescaling/s1K_tokenized \
  --output_dir ./outputs/model_checkpoint \
  --per_device_train_batch_size 8 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --weight_decay 1e-4 \
  --block_size 32768
```

Key features:
- **Flash Attention 2**: Efficient attention for long sequences
- **FSDP Support**: Fully Sharded Data Parallel for large models
- **Completion-only Loss**: Train only on assistant responses
- **DeepSpeed Integration**: Memory-efficient training

#### Training Configurations

- `fsdp_config_qwen.json`: FSDP configuration for Qwen models
- `deepspeed_zero3.yaml`: DeepSpeed ZeRO-3 configuration

#### Freeze Training (Efficient Fine-tuning)

To train only specific modules (e.g., `o_proj`):

1. Modify the DeepSpeed config to specify trainable modules
2. Run training with the modified config:

```bash
# Example: Train only o_proj + norm + lm_head + embed_tokens
accelerate launch --config_file train/fsdp_config_qwen.json \
  train/sft.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --output_dir ./outputs/freeze_oproj \
  ...
```

## Stethoscope Tools

### Delta Stethoscope

**Purpose**: Identify which modules change most during reasoning enhancement

**Key Scripts**:
- `compare/eval.py`: Compute weight differences
- `compare/eval2.py`: Visualize weight distributions

**Example**:
```bash
python compare/eval.py \
  --base_model Qwen/Qwen2.5-32B \
  --reasoning_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

### Merge Stethoscope

**Purpose**: Test if specific modules are sufficient for reasoning

**Key Scripts**:
- `compare/merge.py`: Merge o_proj + norm + head
- `compare/merge2.py`: Merge o_proj only
- `compare/merge2_wo.py`: Merge without specific modules

**Example**:
```bash
python compare/merge.py \
  --base Qwen/Qwen2.5-Math-1.5B \
  --reasoning deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output ./merged_model
```

### Freeze Stethoscope

**Purpose**: Efficient training by freezing non-essential modules

**Key Scripts**:
- DeepSpeed configuration: `deepspeed_zero3.yaml`
- Training script: (can be added based on your training code)

**Results**:
- Training only `o_proj` (+LN) achieves 0.367 on AIME 2024 vs 0.367 for full fine-tuning
- **3x faster** training speed
- Significantly lower GPU memory usage

### Destruction Stethoscope

**Purpose**: Understand module functionality by systematic disabling

**Key Scripts**:
- `compare/rand.py`: Zero out modules
- `compare/rand2.py`: Re-initialize modules
- `compare/rand3.py`: Remove entire layers

**Methods**:
1. **Zero**: Set all parameters to 0
2. **ReInit**: Re-initialize with Gaussian noise
3. **Remove**: Remove entire transformer layers

## Evaluation

The `eval/` directory contains scripts for evaluating models on various benchmarks.

### Running Evaluations

```bash
cd eval

# Run evaluation on AIME 2024
bash srun-eval-s1.sh

# Or use the commands script for various benchmarks
bash commands.sh
```

### Available Evaluation Scripts

- `generate.py`: Generate model responses for evaluation
- `compute_sample_stats.py`: Compute statistics from evaluation results
- `commands.sh`: Collection of evaluation commands for different benchmarks

### Example: Evaluate on Custom Dataset

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# Generate response
prompt = "Solve: What is the derivative of x^2 + 3x + 1?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
```

## Visualization

The `compare/code/` directory contains scripts for visualizing experimental results.

### Available Visualization Tools

- `compare_heatmap.py`: Generate heatmaps for model comparisons
- `compare_heatmap2.py`: Enhanced heatmap with additional metrics
- `compare_heatmap3.py`: Multi-model comparison heatmaps
- `compare_all_percent.py`: Percentage-based comparison visualizations
- `compare_all_abs.py`: Absolute value comparison visualizations
- `compare_single.py`: Single model analysis plots
- `compare_bar.py`: Bar chart comparisons
- `compare_percent2.py`: Advanced percentage visualizations
- `answers.py`: Answer analysis and statistics

### Example: Generate Comparison Heatmap

```bash
cd compare/code
python compare_heatmap.py \
  --model1 ./model_a \
  --model2 ./model_b \
  --output ./results/heatmap.png
```

### Helper Scripts

The `scripts/` directory contains convenience scripts:

- `run_merge_stethoscope.sh`: Run merge stethoscope experiments
- `run_delta_stethoscope.sh`: Run delta stethoscope analysis

### Usage Examples

The `examples/` directory contains example code:

- `merge_stethoscope_example.py`: Example of merging specific modules
- `delta_stethoscope_example.py`: Example of analyzing weight differences

Run examples:
```bash
# Run merge stethoscope example
python examples/merge_stethoscope_example.py

# Or use the helper script
bash scripts/run_merge_stethoscope.sh
```

## Experimental Results

### AIME 2024 Benchmark

| Model | Tuned Modules | Params (B) | Speed (steps/s) | Accuracy |
|-------|---------------|------------|-----------------|----------|
| Qwen2.5-32B (base) | - | - | - | 0.167 |
| + Norm only | 1.5 | 0.055 | 0.200 |
| **+ o_proj (+LN)** | **3.2** | **0.052** | **0.367** |
| + {q,k,v,o}_proj (+LN) | 5.6 | 0.044 | 0.300 |
| + Full model | 32.8 | 0.015 | 0.367 |

### Model Merging Results (1.5B)

| Merged Modules | AIME 2024 | Avg Tokens |
|----------------|-----------|------------|
| Base (Qwen2.5-Math-1.5B) | 0.067 | 2421 |
| **o_proj only** | **0.200** | **5418** |
| {q,k,v}_proj | 0.000 | 2058 |
| MLP | 0.000 | 15532 |
| Reasoning Model (DeepSeek-R1) | 0.233 | 11892 |

## Repository Structure

```
.
├── train/                   # Training scripts
│   ├── sft.py              # Main supervised fine-tuning script
│   ├── sft2.py             # Alternative training implementation
│   ├── sft.sh              # Single-node training script
│   ├── sft_slurm.sh        # SLURM multi-node training
│   ├── sft_accelerate.sh   # Accelerate-based training
│   ├── sft_multinode.sh    # Multi-node training setup
│   └── fsdp_config_qwen.json  # FSDP configuration
├── data/                    # Data processing scripts
│   ├── collect_data.py     # Main data collection pipeline
│   ├── bulk_inference.py   # Batch inference
│   ├── featurization.py    # Feature extraction
│   ├── tokenization.py     # Data tokenization
│   ├── decontaminate_util.py  # Test set decontamination
│   ├── aime.jsonl          # AIME dataset
│   └── utils/              # Utility functions
├── compare/                 # Stethoscope implementations
│   ├── merge.py            # Merge Stethoscope
│   ├── merge2.py           # Alternative merging
│   ├── merge2_wo.py        # Merging without specific modules
│   ├── merge3.py           # Advanced merging
│   ├── eval.py             # Delta Stethoscope
│   ├── eval2.py            # Enhanced evaluation
│   ├── correct.py          # Evaluation helper
│   ├── rand.py             # Zero-out destruction
│   ├── rand2.py            # Re-init destruction
│   ├── rand3.py            # Layer removal destruction
│   ├── rand4.py            # Advanced destruction
│   ├── get_token.py        # Token statistics
│   └── code/               # Visualization scripts
│       ├── compare_heatmap.py    # Heatmap visualizations
│       ├── compare_heatmap2.py   # Enhanced heatmaps
│       ├── compare_heatmap3.py   # Multi-model heatmaps
│       ├── compare_all_percent.py  # Percentage comparisons
│       ├── compare_all_abs.py    # Absolute comparisons
│       ├── compare_single.py     # Single model analysis
│       ├── compare_bar.py        # Bar chart comparisons
│       ├── compare_percent2.py   # Advanced percentages
│       └── answers.py            # Answer analysis
├── eval/                    # Evaluation scripts
│   ├── commands.sh         # Evaluation commands
│   ├── compute_sample_stats.py  # Statistics computation
│   ├── generate.py         # Generation script
│   └── srun-eval-s1.sh     # SLURM evaluation script
├── examples/               # Usage examples
│   ├── merge_stethoscope_example.py  # Merge example
│   └── delta_stethoscope_example.py  # Delta example
├── scripts/                # Helper scripts
│   ├── run_merge_stethoscope.sh  # Run merge experiments
│   └── run_delta_stethoscope.sh  # Run delta analysis
├── assets/                 # Documentation assets
│   └── overview.png        # Overview diagram
├── deepspeed_zero3.yaml    # DeepSpeed configuration
├── deepspeed_zero3-2.yaml  # Alternative DeepSpeed config
├── requirements.txt        # Dependencies
├── run.py                  # Main entry point
├── print.py               # Utility for printing
└── README.md              # This file
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{shao2025who,
  title={Who Reasons in the Large Language Models?},
  author={Shao, Jie and Wu, Jianxin},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work was partly supported by the National Natural Science Foundation of China under Grant 62276123
- We thank the authors of Qwen, DeepSeek, and other open-source models
- Special thanks to Ke Zhu for discussions

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Jie Shao: shaoj@lamda.nju.edu.cn
- Jianxin Wu: wujx2001@nju.edu.cn

---

**National Key Laboratory for Novel Software Technology**
**School of Artificial Intelligence**
**Nanjing University, China**
