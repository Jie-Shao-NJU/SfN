import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
import logging
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import transformers
import trl
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        kwargs = {
            "device_map": "auto", 
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        kwargs = {
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)

    # # Freeze all parameters initially
    # for name, param in model.named_parameters():
    #     param.requires_grad = False

    # Unfreeze only attention output projections
    frozen_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        # if "mlp" in name:
        if "mlp" in name:
            # import pdb; pdb.set_trace()
            param.requires_grad = False
            frozen_params += param.numel()
            print(f"Freezing parameter: {name}")
        else:
            trainable_params += param.numel()

    # Log frozen and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}, Frozen: {frozen_params}")

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    args.report_to = ['tensorboard']

    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
