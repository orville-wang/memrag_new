#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

MEMGEN_ROOT = "/root/autodl-tmp/memrag_new/MemGen"
sys.path.insert(0, MEMGEN_ROOT)
from memgen.model import MemGenModel  # noqa: E402
from memgen.utils import log_trainable_params  # noqa: E402


def build_model_config(args):
    lora = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    return {
        "model_name": args.model,
        "load_model_path": args.init_memgen_path if args.init_memgen_path != "null" else None,
        "attn_implementation": "eager",
        "torch_dtype": "bfloat16",
        "max_prompt_aug_num": args.max_prompt_aug_num,
        "max_inference_aug_num": args.max_inference_aug_num,
        "weaver": {
            "model_name": args.model,
            "prompt_latents_len": args.prompt_latents_len,
            "inference_latents_len": args.inference_latents_len,
            "lora_config": lora,
        },
        "trigger": {
            "model_name": args.model,
            "active": False,
            "lora_config": lora,
        },
    }


def prepare_weaver_sft_params(model):
    """Train only the MemGen weaver path: weaver LoRA/latent params plus projection layers."""
    for _, param in model.named_parameters():
        param.requires_grad = False

    train_name_markers = (
        "lora_A", "lora_B",
        "prompt_query_latents", "inference_query_latents",
        "prompt_latent_ln", "inference_latent_ln",
        "prompt_latent_scale", "inference_latent_scale",
    )
    for name, param in model.weaver.named_parameters():
        if any(marker in name for marker in train_name_markers):
            param.requires_grad = True

    for param in model.reasoner_to_weaver.parameters():
        param.requires_grad = True
    for param in model.weaver_to_reasoner.parameters():
        param.requires_grad = True

    trainable = [(name, param.numel()) for name, param in model.named_parameters() if param.requires_grad]
    total = sum(param.numel() for _, param in model.named_parameters())
    trainable_total = sum(n for _, n in trainable)
    print(f"TRAINABLE_PARAMS {trainable_total} / {total} ({trainable_total / total:.6f})")
    print("TRAINABLE_PARAM_SAMPLE", json.dumps(trainable[:30], ensure_ascii=False))
    return trainable_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-jsonl", default="/root/autodl-tmp/memrag_new/multihop_oracle_sft_n1000/train.jsonl")
    parser.add_argument("--valid-jsonl", default="/root/autodl-tmp/memrag_new/multihop_oracle_sft_n1000/valid.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/hotpot2wiki_weaver_sft_n1000/model")
    parser.add_argument("--init-memgen-path", default="null", help="Optional existing MemGen checkpoint to continue from.")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-aug-num", type=int, default=8)
    parser.add_argument("--max-inference-aug-num", type=int, default=0)
    parser.add_argument("--prompt-latents-len", type=int, default=8)
    parser.add_argument("--inference-latents-len", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    print("ARGS", json.dumps(vars(args), ensure_ascii=False, indent=2))
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))

    ds = load_dataset("json", data_files={"train": args.train_jsonl, "validation": args.valid_jsonl})
    print("DATASET", ds)
    print("TRAIN_SAMPLE", json.dumps(ds["train"][0], ensure_ascii=False)[:2000])

    model = MemGenModel.from_config(build_model_config(args))
    prepare_weaver_sft_params(model)
    log_trainable_params(model)

    train_args = SFTConfig(
        output_dir=args.output_dir,
        logging_dir=str(Path(args.output_dir).parent / "runs"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="adamw_torch",
        bf16=True,
        assistant_only_loss=True,
        max_length=args.max_length,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=["tensorboard"],
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=model.tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    (Path(args.output_dir) / "training_args_snapshot.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAVED_MODEL", args.output_dir)


if __name__ == "__main__":
    main()
