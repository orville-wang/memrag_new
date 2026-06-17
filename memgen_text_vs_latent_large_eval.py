#!/usr/bin/env python3
"""Large-scale visible-text vs latent-experience ablation for MemGen/Search-R1.

This script is intended to run in /root/autodl-tmp/memrag_new on the remote
machine. It imports the existing smoke-test loop and adds two textual baselines:

  - policy_text: generic retrieval-control rules in visible prompt text.
  - visible_experience_text: selected oracle trajectories in visible prompt text.

The latent actor uses the existing MemGen-style hidden experience blend, so all
experience variants share the same selector and oracle trajectory pool.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from flashrag_memgen_multiturn_smoke import (
    SEARCH_R1_URL,
    avg,
    build_initial_prompt,
    doc_hits,
    docs_to_information,
    final_answer_hit,
    load_flashrag_jsonl,
    normalize_text,
    parse_step_action,
    retrieve,
    summarize,
    supporting_titles,
    title_from_doc,
)
from memgen_experience_latent_smoke import (
    ExperienceSelector,
    MemGenExperienceLatentActor,
    MemGenPromptLatentCustomActor,
    format_experience_text,
    load_oracle_experiences,
    question_from_prompt,
)


POLICY_TEXT = """Retrieval policy guidance for the current question.
These guidelines are not evidence. You must still search and use retrieved documents as the evidence path.
1. For comparison questions, search each compared entity separately before answering.
2. For bridge questions, search the bridge entity and the target entity when needed.
3. Avoid repeating a query that has already been searched.
4. Do not answer after only one partial supporting fact when another comparison target remains unchecked."""


def format_visible_experience_text(question: str, examples: List[Dict]) -> str:
    blocks = [
        "Visible retrieval experience for multi-hop QA.",
        "These examples are policy experience, not evidence for the current question. Use them only to decide the next Search or Answer action; current evidence must come from retrieved documents.",
    ]
    for i, ex in enumerate(examples, 1):
        searches = "; ".join(f"S{j + 1}: {q}" for j, q in enumerate(ex["searches"]))
        supports = "; ".join(ex.get("supporting_titles", [])[:4])
        docs = "; ".join(ex.get("doc_titles", [])[:5])
        blocks.append(
            f"Experience {i}:\n"
            f"Question: {ex['question']}\n"
            f"Useful searches: {searches}\n"
            f"Supporting fact targets: {supports}\n"
            f"Retrieved evidence titles: {docs}"
        )
    blocks.append(f"Current question: {question}")
    return "\n\n".join(blocks)


def inject_before_question(prompt: str, visible_text: str) -> str:
    if "Question:" in prompt:
        return prompt.replace("Question:", visible_text.strip() + "\n\nQuestion:", 1)
    return visible_text.strip() + "\n\n" + prompt


class VisiblePolicyTextActor(MemGenPromptLatentCustomActor):
    name = "policy_text"

    def __init__(
        self,
        model_name: str,
        load_path: str,
        max_visible_tokens: int,
        weaver_model_name: Optional[str] = None,
        trigger_model_name: Optional[str] = None,
        prompt_latents_len: int = 8,
        inference_latents_len: int = 8,
        max_prompt_aug_num: int = 8,
        max_inference_aug_num: int = 0,
    ):
        super().__init__(
            model_name,
            load_path,
            weaver_model_name=weaver_model_name,
            trigger_model_name=trigger_model_name,
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
        )
        self.max_visible_tokens = max_visible_tokens
        self.last_memory_text = ""
        self.last_visible_tokens = 0

    def _visible_text(self, prompt: str) -> str:
        return POLICY_TEXT

    def generate(self, prompt: str, max_new_tokens: int, max_input_tokens: int) -> Tuple[str, Optional[List]]:
        visible_text = self._visible_text(prompt)
        visible_text = self.tokenizer.decode(
            self.tokenizer(visible_text, truncation=True, max_length=self.max_visible_tokens)["input_ids"],
            skip_special_tokens=True,
        )
        self.last_memory_text = visible_text
        self.last_visible_tokens = len(self.tokenizer(visible_text)["input_ids"])
        augmented = inject_before_question(prompt, visible_text)
        return self._generate_from_embeds(augmented, max_new_tokens, max_input_tokens)


class VisibleExperienceTextActor(VisiblePolicyTextActor):
    name = "visible_experience_text"

    def __init__(
        self,
        model_name: str,
        load_path: str,
        selector: ExperienceSelector,
        max_visible_tokens: int,
        weaver_model_name: Optional[str] = None,
        trigger_model_name: Optional[str] = None,
        prompt_latents_len: int = 8,
        inference_latents_len: int = 8,
        max_prompt_aug_num: int = 8,
        max_inference_aug_num: int = 0,
    ):
        super().__init__(
            model_name,
            load_path,
            max_visible_tokens,
            weaver_model_name=weaver_model_name,
            trigger_model_name=trigger_model_name,
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
        )
        self.selector = selector

    def _visible_text(self, prompt: str) -> str:
        question = question_from_prompt(prompt)
        examples = self.selector.select(question)
        return format_visible_experience_text(question, examples)


class VisibleExperienceTextCompactActor(VisibleExperienceTextActor):
    name = "visible_experience_text_compact"


class LatentExperienceWithTokenStatsActor(MemGenExperienceLatentActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_visible_tokens = 0

    def _experience_latent(self, prompt: str) -> torch.Tensor:
        latent = super()._experience_latent(prompt)
        self.last_visible_tokens = len(self.tokenizer(self.last_memory_text)["input_ids"]) if self.last_memory_text else 0
        return latent


def query_copy_from_memory(query: str, memory_text: str) -> bool:
    if not query or not memory_text:
        return False
    query_norm = normalize_text(query)
    if len(query_norm) < 8:
        return False
    return query_norm in normalize_text(memory_text)


def run_actor(actor, rows: List[Dict], args) -> List[Dict]:
    import requests

    session = requests.Session()
    out_rows = []
    for idx, item in enumerate(rows):
        support = supporting_titles(item)
        prompt = build_initial_prompt(actor.tokenizer, item["question"])
        trajectory = []
        retrieved_support = set()
        final_answer = ""
        finish_reason = "max_turn"
        repeated_queries = 0
        seen_queries = set()
        first_support_hit_turn = None
        first_answer_hit_turn = None
        all_support_hit_turn = None
        no_hit_searches = 0
        copied_memory_queries = 0
        visible_token_total = 0

        for turn in range(1, args.max_turns + 1):
            step_text, aug_mask = actor.generate(prompt, args.max_new_tokens, args.max_input_tokens)
            memory_text = getattr(actor, "last_memory_text", "")
            visible_tokens = int(getattr(actor, "last_visible_tokens", 0) or 0)
            visible_token_total += visible_tokens
            action_type, action_text = parse_step_action(step_text)
            event = {
                "turn": turn,
                "step_text": step_text,
                "action_type": action_type,
                "action_text": action_text,
                "aug_mask": aug_mask[:1] if aug_mask is not None else None,
                "experience_tokens": visible_tokens,
            }
            if memory_text:
                event["memory_text_preview"] = memory_text[:1000]
            prompt += step_text

            if action_type == "answer":
                final_answer = action_text
                finish_reason = "answer"
                event["final_answer_hit"] = final_answer_hit(final_answer, item.get("golden_answers", []))
                trajectory.append(event)
                break

            if action_type != "search" or not action_text:
                finish_reason = "other"
                trajectory.append(event)
                break

            if query_copy_from_memory(action_text, memory_text):
                copied_memory_queries += 1
                event["query_copied_from_memory"] = True
            else:
                event["query_copied_from_memory"] = False

            qn = normalize_text(action_text)
            if qn in seen_queries:
                repeated_queries += 1
                event["duplicate_query"] = True
            else:
                event["duplicate_query"] = False
                seen_queries.add(qn)

            docs = retrieve(session, action_text, args.topk)
            hits = doc_hits(docs, support, item.get("golden_answers", []))
            for t in hits["hit_support_titles"]:
                retrieved_support.add(normalize_text(t))
            if hits["hit_support_titles"] and first_support_hit_turn is None:
                first_support_hit_turn = turn
            if hits["hit_answer"] and first_answer_hit_turn is None:
                first_answer_hit_turn = turn
            if support and len(retrieved_support) >= len({normalize_text(t) for t in support}) and all_support_hit_turn is None:
                all_support_hit_turn = turn
            if not hits["hit_support_titles"] and not hits["hit_answer"]:
                no_hit_searches += 1

            event.update(hits)
            event["docs"] = [{"id": d.get("id"), "title": title_from_doc(d), "score": d.get("score")} for d in docs]
            trajectory.append(event)
            prompt += docs_to_information(docs, args.max_doc_chars)

        searches = sum(1 for e in trajectory if e["action_type"] == "search")
        support_norm = {normalize_text(t) for t in support}
        summary = {
            "dataset": item["dataset"],
            "id": item.get("id"),
            "question": item["question"],
            "golden_answers": item.get("golden_answers", []),
            "supporting_titles": support,
            "finish_reason": finish_reason,
            "final_answer": final_answer,
            "final_answer_hit": final_answer_hit(final_answer, item.get("golden_answers", [])) if final_answer else False,
            "searches": searches,
            "repeated_queries": repeated_queries,
            "no_hit_searches": no_hit_searches,
            "copied_memory_queries": copied_memory_queries,
            "visible_experience_tokens_total": visible_token_total,
            "visible_experience_tokens_avg_per_turn": visible_token_total / max(1, len(trajectory)),
            "support_coverage": (len(retrieved_support) / len(support_norm)) if support_norm else None,
            "any_support_hit": first_support_hit_turn is not None,
            "first_support_hit_turn": first_support_hit_turn,
            "first_answer_hit_turn": first_answer_hit_turn,
            "all_support_hit_turn": all_support_hit_turn,
            "trajectory": trajectory,
        }
        out_rows.append(summary)
        print(
            actor.name.upper(),
            idx,
            item["dataset"],
            item.get("id"),
            "finish",
            finish_reason,
            "searches",
            searches,
            "dup",
            repeated_queries,
            "nohit",
            no_hit_searches,
            "copy",
            copied_memory_queries,
            "cov",
            summary["support_coverage"],
            "final_hit",
            summary["final_answer_hit"],
            flush=True,
        )
    return out_rows


def summarize_extended(rows: List[Dict]) -> Dict:
    base = summarize(rows)
    searches = sum(r["searches"] for r in rows)
    copied = sum(r.get("copied_memory_queries", 0) for r in rows)
    base.update({
        "avg_copied_memory_queries": avg([r.get("copied_memory_queries", 0) for r in rows]),
        "copy_query_rate_per_search": copied / searches if searches else 0.0,
        "avg_visible_experience_tokens_per_turn": avg([r.get("visible_experience_tokens_avg_per_turn", 0) for r in rows]),
        "finish_reason_counts": dict(sorted({k: sum(1 for r in rows if r.get("finish_reason") == k) for k in {r.get("finish_reason") for r in rows}}.items())),
    })
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--weaver-model", default=None)
    parser.add_argument("--trigger-model", default=None)
    parser.add_argument("--prompt-latents-len", type=int, default=8)
    parser.add_argument("--inference-latents-len", type=int, default=8)
    parser.add_argument("--max-prompt-aug-num", type=int, default=8)
    parser.add_argument("--max-inference-aug-num", type=int, default=0)
    parser.add_argument("--memgen-load-model-path", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/state_experience_blend_sft_a05_steps300/checkpoint-150")
    parser.add_argument("--hotpot-path", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/dev.jsonl")
    parser.add_argument("--twiki-path", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/dev.jsonl")
    parser.add_argument("--oracle-train-path", default="/root/autodl-tmp/memrag_new/multihop_oracle_sft_full_n1000_top3_doc220/train.jsonl")
    parser.add_argument("--num-per-dataset", type=int, default=100)
    parser.add_argument("--experience-pool-size", type=int, default=1000)
    parser.add_argument("--experience-topn", type=int, default=4)
    parser.add_argument("--compact-experience-topn", type=int, default=2)
    parser.add_argument("--max-visible-tokens", type=int, default=1536)
    parser.add_argument("--compact-visible-tokens", type=int, default=512)
    parser.add_argument("--max-memory-tokens", type=int, default=2048)
    parser.add_argument("--max-state-chars", type=int, default=2200)
    parser.add_argument("--experience-alpha", type=float, default=0.5)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    parser.add_argument("--actors", default="custom,policy_text,visible_compact,visible_text,latent", help="Comma-separated: custom,policy_text,visible_compact,visible_text,latent")
    parser.add_argument("--output", default="/root/autodl-tmp/memrag_new/memgen_text_vs_latent_sft150_n200.json")
    args = parser.parse_args()

    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
    print("MODEL", args.model)
    print("SEARCH_R1_URL", SEARCH_R1_URL)
    print("MEMGEN_LOAD_MODEL_PATH", args.memgen_load_model_path)
    print("ORACLE_TRAIN_PATH", args.oracle_train_path)

    data = []
    data.extend(load_flashrag_jsonl(args.hotpot_path, args.num_per_dataset, "hotpotqa"))
    data.extend(load_flashrag_jsonl(args.twiki_path, args.num_per_dataset, "2wikimultihopqa"))
    experiences = load_oracle_experiences(args.oracle_train_path, args.experience_pool_size)
    selector = ExperienceSelector(experiences, args.experience_topn)
    compact_selector = ExperienceSelector(experiences, args.compact_experience_topn)
    print("DATA_N", len(data), "EXPERIENCE_N", len(experiences), "ACTORS", args.actors)

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "Visible-text vs MemGen-style latent experience ablation in FlashRAG/Search-R1 loop",
        "datasets": ["hotpotqa/dev", "2wikimultihopqa/dev"],
        "retriever": "Search-R1 BM25 over wiki-18 via HTTP retrieve API",
        "model": args.model,
        "weaver_model": args.weaver_model or args.model,
        "trigger_model": args.trigger_model or args.model,
        "memgen_load_model_path": args.memgen_load_model_path,
        "settings": vars(args),
        "actors": {},
    }

    actor_names = [x.strip() for x in args.actors.split(",") if x.strip()]
    actor_kwargs = {
        "weaver_model_name": args.weaver_model,
        "trigger_model_name": args.trigger_model,
        "prompt_latents_len": args.prompt_latents_len,
        "inference_latents_len": args.inference_latents_len,
        "max_prompt_aug_num": args.max_prompt_aug_num,
        "max_inference_aug_num": args.max_inference_aug_num,
    }
    for actor_name in actor_names:
        if actor_name == "custom":
            actor = MemGenPromptLatentCustomActor(args.model, args.memgen_load_model_path, **actor_kwargs)
        elif actor_name == "policy_text":
            actor = VisiblePolicyTextActor(args.model, args.memgen_load_model_path, args.max_visible_tokens, **actor_kwargs)
        elif actor_name == "visible_text":
            actor = VisibleExperienceTextActor(args.model, args.memgen_load_model_path, selector, args.max_visible_tokens, **actor_kwargs)
        elif actor_name == "visible_compact":
            actor = VisibleExperienceTextCompactActor(args.model, args.memgen_load_model_path, compact_selector, args.compact_visible_tokens, **actor_kwargs)
        elif actor_name == "latent":
            actor = LatentExperienceWithTokenStatsActor(
                args.model,
                args.memgen_load_model_path,
                selector,
                args.max_memory_tokens,
                args.max_state_chars,
                args.experience_alpha,
                **actor_kwargs,
            )
        else:
            raise ValueError(f"unknown actor: {actor_name}")
        print("RUN_ACTOR", actor.name, "LORA_COMPARE_MAX_ABS_DIFF", actor.lora_diff)
        rows = run_actor(actor, data, args)
        result["actors"][actor.name] = {
            "summary": summarize_extended(rows),
            "lora_compare_max_abs_diff": actor.lora_diff,
            "rows": rows,
        }
        result["summary"] = {k: v["summary"] for k, v in result["actors"].items()}
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        actor.close()
        gc.collect()
        torch.cuda.empty_cache()

    result["summary"] = {k: v["summary"] for k, v in result["actors"].items()}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY_JSON", json.dumps(result["summary"], ensure_ascii=False))
    print("RESULT_PATH", args.output)


if __name__ == "__main__":
    main()
