#!/usr/bin/env python3
"""Frozen-parameter memory-carrier policy fidelity experiment.

This script follows docs/experiment_design_v0.1.md through M0-M2.  It does not
train and does not run Search-R1.  It freezes the base reasoner and changes only
the carrier of the same memory ``m``: visible text, compressed text, MemGen-style
latent tokens, random latent tokens, or fixed soft prompt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMGEN_ROOT = REPO_ROOT / "MemGen"
if str(MEMGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMGEN_ROOT))

from memgen.model import MemGenModel  # noqa: E402

from memory_carriers import (  # noqa: E402
    SummaryCache,
    build_full_memory,
    extractive_memory,
    fixed_soft_prompt_latents,
    generated_summary_memory,
    keywords,
    make_action_candidates,
    mean_pool_latents,
    random_latents_like,
    stable_hash,
    truncate_to_tokens,
)
from memgen_injector import inject_after_prompt, make_position_ids  # noqa: E402


TEXT_ARMS = {"no_memory", "full_text", "extractive_text", "summary_text"}
LATENT_ARMS = {"latent_compressed", "random_latent", "fixed_soft_prompt"}
ALL_ARMS = ["no_memory", "full_text", "extractive_text", "summary_text", "latent_compressed", "random_latent", "fixed_soft_prompt"]


def fixture_samples() -> List[Dict]:
    return [
        {
            "id": "fixture_trivia_0",
            "dataset": "fixture_trivia",
            "question": "Who wrote the novel Pride and Prejudice?",
            "gold": "Jane Austen",
            "wrong": "Charlotte Bronte",
        },
        {
            "id": "fixture_trivia_1",
            "dataset": "fixture_trivia",
            "question": "What city is the Eiffel Tower located in?",
            "gold": "Paris",
            "wrong": "Rome",
        },
        {
            "id": "fixture_math_0",
            "dataset": "fixture_gsm8k",
            "question": "A shop has 12 apples and sells 5. How many apples remain?",
            "gold": "7",
            "wrong": "17",
        },
        {
            "id": "fixture_math_1",
            "dataset": "fixture_gsm8k",
            "question": "If 3 bags hold 4 marbles each, how many marbles are there?",
            "gold": "12",
            "wrong": "7",
        },
        {
            "id": "fixture_trivia_2",
            "dataset": "fixture_trivia",
            "question": "Which planet is known as the Red Planet?",
            "gold": "Mars",
            "wrong": "Venus",
        },
    ]


def parse_gsm_answer(answer: str) -> str:
    if "####" in answer:
        return answer.split("####")[-1].strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", answer)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
    return nums[-1] if nums else answer.strip()[:80]


def load_hf_samples(limit_per_dataset: int, datasets: List[str]) -> Tuple[List[Dict], List[str]]:
    warnings = []
    samples: List[Dict] = []
    try:
        from datasets import load_dataset
    except Exception as exc:
        return fixture_samples(), [f"datasets import failed: {exc}; using fixture samples"]

    if "triviaqa" in datasets:
        try:
            ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split=f"validation[:{limit_per_dataset}]")
            for i, row in enumerate(ds):
                aliases = row.get("answer", {}).get("normalized_aliases") or row.get("answer", {}).get("aliases") or []
                gold = aliases[0] if aliases else row.get("answer", {}).get("value", "")
                if gold:
                    samples.append({
                        "id": f"triviaqa_{i}",
                        "dataset": "triviaqa",
                        "question": row["question"],
                        "gold": str(gold),
                    })
        except Exception as exc:
            warnings.append(f"triviaqa load failed: {exc}")

    if "gsm8k" in datasets:
        try:
            ds = load_dataset("gsm8k", "main", split=f"test[:{limit_per_dataset}]")
            for i, row in enumerate(ds):
                samples.append({
                    "id": f"gsm8k_{i}",
                    "dataset": "gsm8k",
                    "question": row["question"],
                    "gold": parse_gsm_answer(row["answer"]),
                })
        except Exception as exc:
            warnings.append(f"gsm8k load failed: {exc}")

    if not samples:
        warnings.append("all requested datasets failed; using fixture samples")
        samples = fixture_samples()

    # Fill distractors from a different sample whenever possible.
    golds = [s["gold"] for s in samples]
    for i, sample in enumerate(samples):
        sample["wrong"] = next((g for j, g in enumerate(golds) if j != i and g != sample["gold"]), "I don't know")
    return samples, warnings


def build_prompt(tokenizer, question: str, memory_text: Optional[str]) -> str:
    content = (
        "Choose the next action for the task. Valid action forms are:\n"
        "Answer: <short answer>\n"
        "Search: <search query>\n\n"
    )
    if memory_text:
        content += f"Memory:\n{memory_text}\n\n"
    content += f"Question: {question}\nNext action:"
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return content


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> List[Dict]:
    probs = F.softmax(logits.float(), dim=-1)
    vals, idxs = torch.topk(probs, k)
    out = []
    for prob, idx in zip(vals.tolist(), idxs.tolist()):
        out.append({"id": int(idx), "text": tokenizer.decode([idx]), "prob": float(prob)})
    return out


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p_log = F.log_softmax(p_logits.float(), dim=-1)
    q_log = F.log_softmax(q_logits.float(), dim=-1)
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum().item())


def topk_overlap(a_logits: torch.Tensor, b_logits: torch.Tensor, k: int) -> float:
    a = set(torch.topk(a_logits, k).indices.tolist())
    b = set(torch.topk(b_logits, k).indices.tolist())
    return len(a & b) / max(1, k)


def candidate_policy(scores: Dict[str, float]) -> Dict[str, float]:
    ids = list(scores)
    vals = torch.tensor([scores[i] for i in ids], dtype=torch.float32)
    probs = F.softmax(vals, dim=0)
    return {i: float(p) for i, p in zip(ids, probs.tolist())}


def kl_dict(p: Dict[str, float], q: Dict[str, float]) -> float:
    out = 0.0
    for key, p_val in p.items():
        q_val = max(q.get(key, 0.0), 1e-12)
        p_val = max(p_val, 1e-12)
        out += p_val * math.log(p_val / q_val)
    return float(out)


def score_candidate_sequence(
    reasoner,
    tokenizer,
    prompt: str,
    candidate: str,
    latents: Optional[torch.Tensor],
    max_prompt_tokens: int,
) -> Tuple[float, float, int]:
    device = reasoner.device
    encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_prompt_tokens).to(device)
    candidate_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if candidate_ids.numel() == 0:
        raise ValueError(f"empty candidate after tokenization: {candidate!r}")

    prompt_embeds = reasoner.get_input_embeddings()(encoded_prompt["input_ids"])
    injected = inject_after_prompt(prompt_embeds, encoded_prompt["attention_mask"], latents)
    candidate_embeds = reasoner.get_input_embeddings()(candidate_ids)
    inputs_embeds = torch.cat([injected.inputs_embeds, candidate_embeds], dim=1)
    candidate_attention = torch.ones(candidate_ids.shape, dtype=injected.attention_mask.dtype, device=device)
    attention_mask = torch.cat([injected.attention_mask, candidate_attention], dim=1)
    position_ids = make_position_ids(attention_mask)
    start = injected.inputs_embeds.size(1)

    with torch.no_grad():
        outputs = reasoner(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    logits = outputs.logits[0]
    token_ids = candidate_ids[0]
    logps = []
    for j, token_id in enumerate(token_ids):
        pos = start + j - 1
        logp = F.log_softmax(logits[pos].float(), dim=-1)[token_id]
        logps.append(logp)
    total = torch.stack(logps).sum()
    avg = torch.stack(logps).mean()
    return float(total.item()), float(avg.item()), int(candidate_ids.size(1))


def score_next_token(reasoner, tokenizer, prompt: str, latents: Optional[torch.Tensor], max_prompt_tokens: int) -> torch.Tensor:
    device = reasoner.device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_prompt_tokens).to(device)
    prompt_embeds = reasoner.get_input_embeddings()(encoded["input_ids"])
    injected = inject_after_prompt(prompt_embeds, encoded["attention_mask"], latents)
    with torch.no_grad():
        outputs = reasoner(
            inputs_embeds=injected.inputs_embeds,
            attention_mask=injected.attention_mask,
            position_ids=injected.position_ids,
            use_cache=False,
        )
    return outputs.logits[0, -1].detach().float().cpu()


def score_arm(
    memgen_model,
    tokenizer,
    sample: Dict,
    arm: str,
    memory_texts: Dict[str, str],
    latents_by_arm: Dict[str, torch.Tensor],
    candidates: List[Dict[str, str]],
    args,
) -> Tuple[Dict, torch.Tensor, Dict[str, float]]:
    if arm == "no_memory":
        prompt = build_prompt(tokenizer, sample["question"], None)
        latents = None
        carrier_tokens = 0
        latent_count = 0
    elif arm in TEXT_ARMS:
        prompt = build_prompt(tokenizer, sample["question"], memory_texts[arm])
        latents = None
        carrier_tokens = len(tokenizer(memory_texts[arm], add_special_tokens=False)["input_ids"])
        latent_count = 0
    elif arm in LATENT_ARMS:
        prompt = build_prompt(tokenizer, sample["question"], None)
        latents = latents_by_arm[arm]
        carrier_tokens = 0
        latent_count = int(latents.size(1))
    else:
        raise ValueError(f"unknown arm: {arm}")

    next_logits = score_next_token(memgen_model.reasoner, tokenizer, prompt, latents, args.max_prompt_tokens)
    candidate_totals = {}
    candidate_avgs = {}
    candidate_lens = {}
    for cand in candidates:
        total, avg, length = score_candidate_sequence(
            memgen_model.reasoner,
            tokenizer,
            prompt,
            cand["text"],
            latents,
            args.max_prompt_tokens,
        )
        candidate_totals[cand["id"]] = total
        candidate_avgs[cand["id"]] = avg
        candidate_lens[cand["id"]] = length
    policy = candidate_policy(candidate_avgs)
    non_gold = [v for k, v in candidate_avgs.items() if k != "gold"]
    gold_margin = candidate_avgs["gold"] - max(non_gold)
    row = {
        "carrier_tokens": carrier_tokens,
        "latent_count": latent_count,
        "top_tokens": top_tokens(tokenizer, next_logits, args.top_tokens),
        "candidate_total_logprobs": candidate_totals,
        "candidate_avg_logprobs": candidate_avgs,
        "candidate_lengths": candidate_lens,
        "candidate_policy": policy,
        "gold_candidate_prob": policy["gold"],
        "gold_vs_best_non_gold_margin": float(gold_margin),
    }
    return row, next_logits, policy


def load_memgen_model(args):
    lora = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_rank * 2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    config = {
        "model_name": args.model,
        "load_model_path": args.memgen_load_model_path,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "max_prompt_aug_num": 1,
        "max_inference_aug_num": 0,
        "weaver": {"model_name": args.weaver_model or args.model, "prompt_latents_len": args.latent_len, "inference_latents_len": args.latent_len, "lora_config": lora},
        "trigger": {"model_name": args.trigger_model or args.model, "active": False, "lora_config": lora},
    }
    model = MemGenModel.from_config(config).eval()
    if torch.cuda.is_available() and args.device == "cuda":
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model


def build_sample_carriers(memgen_model, tokenizer, sample: Dict, summary_cache: SummaryCache, args) -> Tuple[Dict[str, str], Dict[str, torch.Tensor]]:
    full = build_full_memory(sample)
    memory_texts = {
        "full_text": full,
        "extractive_text": extractive_memory(tokenizer, full, args.compressed_text_tokens),
    }
    if args.disable_generated_summary:
        memory_texts["summary_text"] = truncate_to_tokens(tokenizer, memory_texts["extractive_text"], args.compressed_text_tokens)
    else:
        memory_texts["summary_text"] = generated_summary_memory(
            memgen_model.reasoner,
            tokenizer,
            full,
            args.compressed_text_tokens,
            cache=summary_cache,
            max_new_tokens=args.summary_max_new_tokens,
        )

    dtype = memgen_model.reasoner.get_input_embeddings().weight.dtype
    device = memgen_model.device
    latent = mean_pool_latents(memgen_model.reasoner, tokenizer, full, args.latent_len, device=device, dtype=dtype)
    seed = int(stable_hash(sample["id"])[:8], 16) + args.seed
    latents_by_arm = {
        "latent_compressed": latent,
        "random_latent": random_latents_like(latent, seed=seed),
        "fixed_soft_prompt": fixed_soft_prompt_latents(memgen_model, args.latent_len, dtype=dtype),
    }
    return memory_texts, latents_by_arm


def run(args) -> Dict:
    torch.manual_seed(args.seed)
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
    print("MODEL", args.model)
    print("MODE", args.mode, "ARMS", args.arms)
    model = load_memgen_model(args)
    tokenizer = model.tokenizer

    if args.mode == "m0":
        samples = fixture_samples()
        warnings = []
    else:
        samples, warnings = load_hf_samples(args.limit_per_dataset, [d.strip() for d in args.datasets.split(",") if d.strip()])
    if args.limit_total:
        samples = samples[: args.limit_total]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_cache = SummaryCache(args.summary_cache)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "model": args.model,
        "weaver_model": args.weaver_model or args.model,
        "trigger_model": args.trigger_model or args.model,
        "samples": len(samples),
        "arms": arms,
        "settings": vars(args),
        "warnings": warnings,
        "schema": "one JSON object per sample; arms are nested under row['arms']",
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with output_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            memory_texts, latents_by_arm = build_sample_carriers(model, tokenizer, sample, summary_cache, args)
            candidates = make_action_candidates(sample, sample.get("wrong", "I don't know"))

            arm_rows: Dict[str, Dict] = {}
            logits_by_arm: Dict[str, torch.Tensor] = {}
            candidate_policy_by_arm: Dict[str, Dict[str, float]] = {}
            for arm in arms:
                arm_row, logits, cand_policy = score_arm(
                    model,
                    tokenizer,
                    sample,
                    arm,
                    memory_texts,
                    latents_by_arm,
                    candidates,
                    args,
                )
                arm_rows[arm] = arm_row
                logits_by_arm[arm] = logits
                candidate_policy_by_arm[arm] = cand_policy

            comparisons = {}
            full_logits = logits_by_arm.get("full_text")
            no_logits = logits_by_arm.get("no_memory")
            full_policy = candidate_policy_by_arm.get("full_text")
            no_policy = candidate_policy_by_arm.get("no_memory")
            no_margin = arm_rows.get("no_memory", {}).get("gold_vs_best_non_gold_margin")
            full_margin = arm_rows.get("full_text", {}).get("gold_vs_best_non_gold_margin")
            denom = None
            if no_margin is not None and full_margin is not None and abs(full_margin - no_margin) > 1e-8:
                denom = full_margin - no_margin

            for arm in arms:
                comp = {}
                if full_logits is not None:
                    comp["next_token_kl_full_to_arm"] = kl_from_logits(full_logits, logits_by_arm[arm])
                    comp["next_token_top10_overlap_with_full"] = topk_overlap(full_logits, logits_by_arm[arm], 10)
                if no_logits is not None:
                    comp["next_token_kl_arm_to_no"] = kl_from_logits(logits_by_arm[arm], no_logits)
                if full_policy is not None:
                    comp["candidate_kl_full_to_arm"] = kl_dict(full_policy, candidate_policy_by_arm[arm])
                if no_policy is not None:
                    comp["candidate_kl_arm_to_no"] = kl_dict(candidate_policy_by_arm[arm], no_policy)
                if denom is not None:
                    comp["margin_recovery_ratio"] = (arm_rows[arm]["gold_vs_best_non_gold_margin"] - no_margin) / denom
                comparisons[arm] = comp

            row = {
                "idx": idx,
                "id": sample["id"],
                "dataset": sample["dataset"],
                "question": sample["question"],
                "gold": sample["gold"],
                "wrong": sample.get("wrong"),
                "memory_hash": stable_hash(memory_texts["full_text"]),
                "memory_preview": memory_texts["full_text"][:400],
                "candidates": candidates,
                "arms": arm_rows,
                "comparisons": comparisons,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                "ROW",
                idx,
                sample["dataset"],
                sample["id"],
                "gold",
                sample["gold"],
                "full_margin",
                arm_rows.get("full_text", {}).get("gold_vs_best_non_gold_margin"),
                flush=True,
            )

    print("RESULT_PATH", output_path)
    print("META_PATH", meta_path)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["m0", "m1", "m2", "all"], default="m0")
    parser.add_argument("--model", default="llamafactory/tiny-random-qwen2.5")
    parser.add_argument("--weaver-model", default=None)
    parser.add_argument("--trigger-model", default=None)
    parser.add_argument("--memgen-load-model-path", default=None)
    parser.add_argument("--arms", default=",".join(ALL_ARMS))
    parser.add_argument("--datasets", default="triviaqa,gsm8k")
    parser.add_argument("--limit-per-dataset", type=int, default=50)
    parser.add_argument("--limit-total", type=int, default=0)
    parser.add_argument("--latent-len", type=int, default=8)
    parser.add_argument("--compressed-text-tokens", type=int, default=8)
    parser.add_argument("--summary-max-new-tokens", type=int, default=48)
    parser.add_argument("--disable-generated-summary", action="store_true")
    parser.add_argument("--summary-cache", default="results/policy_fidelity/cache/summary_cache.json")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--top-tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", default="results/policy_fidelity/m0_tiny_policy.jsonl")
    args = parser.parse_args()

    if args.mode == "m1" and args.arms == ",".join(ALL_ARMS):
        args.arms = "no_memory,full_text,extractive_text,summary_text"
    elif args.mode == "m2" and args.arms == ",".join(ALL_ARMS):
        args.arms = "no_memory,full_text,latent_compressed,random_latent,fixed_soft_prompt"

    run(args)


if __name__ == "__main__":
    main()
