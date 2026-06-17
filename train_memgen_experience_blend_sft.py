#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

MEMGEN_ROOT = "/root/autodl-tmp/memrag_new/MemGen"
sys.path.insert(0, MEMGEN_ROOT)
from memgen.model import MemGenModel  # noqa: E402


def read_jsonl(path: str, limit: int = 0) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


class StateExperienceDataset(Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


def collate_one(batch: List[Dict]) -> Dict:
    assert len(batch) == 1
    return batch[0]


def build_model_config(args) -> Dict:
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
        "max_prompt_aug_num": 8,
        "max_inference_aug_num": 0,
        "weaver": {"model_name": args.model, "prompt_latents_len": 8, "inference_latents_len": 8, "lora_config": lora},
        "trigger": {"model_name": args.model, "active": False, "lora_config": lora},
    }


def prepare_trainable_params(model: MemGenModel) -> int:
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
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    trainable_total = sum(n for _, n in trainable)
    total = sum(p.numel() for p in model.parameters())
    print(f"TRAINABLE_PARAMS {trainable_total} / {total} ({trainable_total / total:.6f})")
    print("TRAINABLE_SAMPLE", json.dumps(trainable[:30], ensure_ascii=False))
    return trainable_total


def encode_truncate(tokenizer, text: str, max_tokens: int, keep_head_ratio: float = 0.35) -> torch.LongTensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_tokens and len(ids) > max_tokens:
        head = int(max_tokens * keep_head_ratio)
        tail = max_tokens - head
        ids = ids[:head] + ids[-tail:]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def prompt_aug_latent(model: MemGenModel, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = model._generate_position_ids(attention_mask)
    weaver_inputs = model.reasoner_to_weaver(inputs_embeds)
    weaver_hidden, _, _ = model.weaver.augment_prompt(weaver_inputs, attention_mask, position_ids)
    return model.weaver_to_reasoner(weaver_hidden)


def forward_loss(model: MemGenModel, tokenizer, sample: Dict, args) -> Tuple[torch.Tensor, Dict]:
    device = model.device
    prompt_ids = encode_truncate(tokenizer, sample["prompt"], args.max_prompt_tokens).to(device)
    memory_ids = encode_truncate(tokenizer, sample["experience_text"], args.max_memory_tokens).to(device)
    target_ids = encode_truncate(tokenizer, sample["target_text"], args.max_target_tokens, keep_head_ratio=1.0).to(device)

    prompt_mask = torch.ones_like(prompt_ids, device=device)
    memory_mask = torch.ones_like(memory_ids, device=device)
    target_mask = torch.ones_like(target_ids, device=device)

    embed = model.reasoner.get_input_embeddings()
    prompt_embeds = embed(prompt_ids)
    memory_embeds = embed(memory_ids)
    target_embeds = embed(target_ids)

    native_latent = prompt_aug_latent(model, prompt_embeds, prompt_mask)
    experience_latent = prompt_aug_latent(model, memory_embeds, memory_mask)
    blended_latent = native_latent * (1.0 - args.experience_alpha) + experience_latent * args.experience_alpha

    inputs_embeds = torch.cat([prompt_embeds, blended_latent, target_embeds], dim=1)
    latent_mask = torch.ones(blended_latent.shape[:2], dtype=prompt_mask.dtype, device=device)
    attention_mask = torch.cat([prompt_mask, latent_mask, target_mask], dim=1)
    position_ids = model._generate_position_ids(attention_mask)

    labels = torch.full(attention_mask.shape, -100, dtype=torch.long, device=device)
    target_start = prompt_ids.size(1) + blended_latent.size(1)
    labels[:, target_start:] = target_ids

    outputs = model.reasoner(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    with torch.no_grad():
        mask = shift_labels != -100
        if mask.any():
            pred = shift_logits.argmax(dim=-1)
            acc = (pred[mask] == shift_labels[mask]).float().mean().item()
            ntok = int(mask.sum().item())
        else:
            acc = 0.0
            ntok = 0
    stats = {
        "loss": float(loss.detach().cpu()),
        "token_acc": acc,
        "target_tokens": ntok,
        "prompt_tokens": int(prompt_ids.size(1)),
        "memory_tokens": int(memory_ids.size(1)),
        "action_type": sample.get("action_type"),
    }
    return loss, stats


@torch.no_grad()
def evaluate(model: MemGenModel, tokenizer, rows: List[Dict], args, max_samples: int) -> Dict:
    model.eval()
    eval_rows = rows[:max_samples] if max_samples else rows
    losses, accs, target_tokens = [], [], []
    by_action = {}
    for sample in tqdm(eval_rows, desc="eval", leave=False):
        loss, stats = forward_loss(model, tokenizer, sample, args)
        losses.append(stats["loss"])
        accs.append(stats["token_acc"])
        target_tokens.append(stats["target_tokens"])
        typ = stats["action_type"]
        by_action.setdefault(typ, {"loss": [], "acc": [], "n": 0})
        by_action[typ]["loss"].append(stats["loss"])
        by_action[typ]["acc"].append(stats["token_acc"])
        by_action[typ]["n"] += 1
    out = {
        "n": len(eval_rows),
        "loss": sum(losses) / len(losses) if losses else None,
        "token_acc": sum(accs) / len(accs) if accs else None,
        "avg_target_tokens": sum(target_tokens) / len(target_tokens) if target_tokens else None,
        "by_action": {},
    }
    for typ, vals in by_action.items():
        out["by_action"][typ] = {
            "n": vals["n"],
            "loss": sum(vals["loss"]) / len(vals["loss"]),
            "token_acc": sum(vals["acc"]) / len(vals["acc"]),
        }
    model.train()
    return out


def save_eval(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--init-memgen-path", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model")
    parser.add_argument("--train-jsonl", default="/root/autodl-tmp/memrag_new/memgen_state_experience_sft_n1000/train.jsonl")
    parser.add_argument("--valid-jsonl", default="/root/autodl-tmp/memrag_new/memgen_state_experience_sft_n1000/valid.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/state_experience_blend_sft_a05_steps300/model")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=150)
    parser.add_argument("--eval-samples", type=int, default=120)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--valid-limit", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-memory-tokens", type=int, default=1024)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--experience-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    print("ARGS", json.dumps(vars(args), ensure_ascii=False, indent=2))
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))

    train_rows = read_jsonl(args.train_jsonl, args.train_limit)
    valid_rows = read_jsonl(args.valid_jsonl, args.valid_limit)
    print("DATA", {"train": len(train_rows), "valid": len(valid_rows)})
    print("TRAIN_SAMPLE", json.dumps(train_rows[0], ensure_ascii=False)[:1600] if train_rows else "null")

    model = MemGenModel.from_config(build_model_config(args)).to(torch.bfloat16).cuda()
    tokenizer = model.tokenizer
    prepare_trainable_params(model)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "training_args_snapshot.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    initial_eval = evaluate(model, tokenizer, valid_rows, args, args.eval_samples)
    print("EVAL_INITIAL", json.dumps(initial_eval, ensure_ascii=False))
    eval_history = [{"step": 0, "eval": initial_eval}]
    save_eval(out_dir / "eval_history.json", eval_history)

    data_loader = DataLoader(StateExperienceDataset(train_rows), batch_size=1, shuffle=True, collate_fn=collate_one)
    step = 0
    micro = 0
    running = []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.max_steps, desc="train_steps")
    epoch = 0
    while step < args.max_steps:
        epoch += 1
        for sample in data_loader:
            loss, stats = forward_loss(model, tokenizer, sample, args)
            scaled = loss / args.gradient_accumulation_steps
            scaled.backward()
            running.append(stats)
            micro += 1
            if micro % args.gradient_accumulation_steps == 0:
                if args.warmup_steps and step < args.warmup_steps:
                    lr_scale = (step + 1) / args.warmup_steps
                    for group in optimizer.param_groups:
                        group["lr"] = args.learning_rate * lr_scale
                else:
                    for group in optimizer.param_groups:
                        group["lr"] = args.learning_rate
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)

                if step % args.logging_steps == 0:
                    window = running[-args.logging_steps * args.gradient_accumulation_steps:]
                    msg = {
                        "step": step,
                        "epoch": epoch,
                        "loss": sum(x["loss"] for x in window) / len(window),
                        "token_acc": sum(x["token_acc"] for x in window) / len(window),
                        "avg_prompt_tokens": sum(x["prompt_tokens"] for x in window) / len(window),
                        "avg_memory_tokens": sum(x["memory_tokens"] for x in window) / len(window),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    print("TRAIN_LOG", json.dumps(msg, ensure_ascii=False))

                if step % args.eval_steps == 0:
                    eval_result = evaluate(model, tokenizer, valid_rows, args, args.eval_samples)
                    print("EVAL", json.dumps({"step": step, "eval": eval_result}, ensure_ascii=False))
                    eval_history.append({"step": step, "eval": eval_result})
                    save_eval(out_dir / "eval_history.json", eval_history)
                    gc.collect()
                    torch.cuda.empty_cache()

                if args.save_steps and step % args.save_steps == 0:
                    ckpt_dir = out_dir.parent / f"checkpoint-{step}"
                    model.save_pretrained(str(ckpt_dir))
                    print("SAVED_CHECKPOINT", str(ckpt_dir))

                if step >= args.max_steps:
                    break
        if step >= args.max_steps:
            break
    pbar.close()

    final_eval = evaluate(model, tokenizer, valid_rows, args, args.eval_samples)
    print("EVAL_FINAL", json.dumps(final_eval, ensure_ascii=False))
    eval_history.append({"step": step, "eval": final_eval})
    save_eval(out_dir / "eval_history.json", eval_history)
    model.save_pretrained(str(out_dir))
    print("SAVED_MODEL", str(out_dir))


if __name__ == "__main__":
    main()
