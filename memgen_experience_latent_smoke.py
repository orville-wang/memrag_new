#!/usr/bin/env python3
import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import GenerationConfig

from flashrag_memgen_multiturn_smoke import (
    SEARCH_R1_URL,
    MemGenActor,
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
    truncate_at_first_stop,
    truncate_prompt,
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "had",
    "has", "have", "he", "her", "his", "in", "is", "it", "its", "of", "on", "or", "she", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "with", "you", "your", "question", "answer",
}


def extract_searches(messages: List[Dict]) -> List[str]:
    searches = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for m in re.finditer(r"<search>(.*?)</search>", msg.get("content", ""), flags=re.S | re.I):
            q = re.sub(r"\s+", " ", m.group(1)).strip()
            if q:
                searches.append(q)
    return searches


def extract_doc_titles(messages: List[Dict], max_titles: int = 6) -> List[str]:
    titles = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for m in re.finditer(r"Doc\s+\d+\(Title:\s*([^\)]+)\)", msg.get("content", ""), flags=re.I):
            title = re.sub(r"\s+", " ", m.group(1)).strip()
            if title and title not in titles:
                titles.append(title)
            if len(titles) >= max_titles:
                return titles
    return titles


def load_oracle_experiences(path: str, limit: int) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            searches = extract_searches(row.get("messages", []))
            if len(searches) < 2:
                continue
            text_terms = set(normalize_text(row.get("question", "")).split()) - STOP_WORDS
            rows.append({
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "question": row.get("question", ""),
                "golden_answers": row.get("golden_answers", []),
                "supporting_titles": row.get("supporting_titles", []),
                "searches": searches[:4],
                "doc_titles": extract_doc_titles(row.get("messages", []), max_titles=6),
                "terms": text_terms,
            })
            if limit and len(rows) >= limit:
                break
    return rows


def question_from_prompt(prompt: str) -> str:
    matches = re.findall(r"Question:\s*(.*?)(?:\n|<\|im_end\|>|$)", prompt, flags=re.S)
    return re.sub(r"\s+", " ", matches[0]).strip() if matches else ""


class ExperienceSelector:
    def __init__(self, rows: List[Dict], topn: int):
        self.rows = rows
        self.topn = topn
        self.cache: Dict[str, List[Dict]] = {}

    def select(self, question: str) -> List[Dict]:
        key = normalize_text(question)
        if key in self.cache:
            return self.cache[key]
        q_terms = set(key.split()) - STOP_WORDS
        scored = []
        for row in self.rows:
            overlap = len(q_terms & row["terms"])
            phrase = 1 if row["question"] and normalize_text(row["question"]) in key else 0
            score = overlap + phrase * 3
            if score > 0:
                scored.append((score, row))
        if not scored:
            scored = [(0, row) for row in self.rows[: self.topn]]
        scored.sort(key=lambda x: (-x[0], x[1].get("id") or ""))
        selected = [row for _, row in scored[: self.topn]]
        self.cache[key] = selected
        return selected


def format_experience_text(question: str, prompt: str, examples: List[Dict], max_prompt_chars: int) -> str:
    blocks = [
        "Retrieval experience memory for multi-hop QA.",
        "Use these successful trajectories as policy experience: search bridge entities, cover distinct supporting facts, avoid duplicate searches, and do not answer before enough evidence is found.",
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
    state_tail = prompt[-max_prompt_chars:]
    blocks.append(
        "Current retrieval state that the hidden memory should guide:\n"
        f"Question: {question}\n"
        f"State tail:\n{state_tail}"
    )
    return "\n\n".join(blocks)


class MemGenPromptLatentCustomActor(MemGenActor):
    name = "memgen_prompt_latent_custom"

    def _prompt_aug_latent(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = self.model._generate_position_ids(attention_mask)
        weaver_inputs_embeds = self.model.reasoner_to_weaver(inputs_embeds)
        weaver_hidden_states, _, _ = self.model.weaver.augment_prompt(
            weaver_inputs_embeds,
            attention_mask,
            position_ids,
        )
        return self.model.weaver_to_reasoner(weaver_hidden_states)

    def _generate_from_embeds(
        self,
        prompt: str,
        max_new_tokens: int,
        max_input_tokens: int,
        blend_latent: Optional[torch.Tensor] = None,
        blend_alpha: float = 0.0,
    ) -> Tuple[str, Optional[List]]:
        prompt = truncate_prompt(self.tokenizer, prompt, max_input_tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            inputs_embeds = self.model.reasoner.get_input_embeddings()(input_ids)
            native_latent = self._prompt_aug_latent(inputs_embeds, attention_mask)
            if blend_latent is not None:
                blend_latent = blend_latent.to(device=native_latent.device, dtype=native_latent.dtype)
                native_latent = native_latent * (1.0 - blend_alpha) + blend_latent * blend_alpha

            native_latent = native_latent.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            inputs_embeds = torch.cat([inputs_embeds, native_latent], dim=1)
            latent_mask = torch.ones(native_latent.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, latent_mask], dim=1)

            gen_cfg = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )
            generated = self.model.reasoner.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
            )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return truncate_at_first_stop(text), None

    def generate(self, prompt: str, max_new_tokens: int, max_input_tokens: int) -> Tuple[str, Optional[List]]:
        return self._generate_from_embeds(prompt, max_new_tokens, max_input_tokens)


class MemGenExperienceLatentActor(MemGenPromptLatentCustomActor):
    name = "memgen_experience_latent"

    def __init__(
        self,
        model_name: str,
        load_path: str,
        selector: ExperienceSelector,
        max_memory_tokens: int,
        max_state_chars: int,
        experience_alpha: float,
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
        self.selector = selector
        self.max_memory_tokens = max_memory_tokens
        self.max_state_chars = max_state_chars
        self.experience_alpha = experience_alpha
        self.name = f"memgen_experience_blend_a{experience_alpha:g}"
        self.last_memory_text = ""

    def _experience_latent(self, prompt: str) -> torch.Tensor:
        question = question_from_prompt(prompt)
        examples = self.selector.select(question)
        memory_text = format_experience_text(question, prompt, examples, self.max_state_chars)
        self.last_memory_text = memory_text
        memory_text = truncate_prompt(self.tokenizer, memory_text, self.max_memory_tokens)
        inputs = self.tokenizer(memory_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            embeds = self.model.reasoner.get_input_embeddings()(inputs["input_ids"])
            return self._prompt_aug_latent(embeds, inputs["attention_mask"])

    def generate(self, prompt: str, max_new_tokens: int, max_input_tokens: int) -> Tuple[str, Optional[List]]:
        exp_latent = self._experience_latent(prompt)
        return self._generate_from_embeds(
            prompt,
            max_new_tokens,
            max_input_tokens,
            blend_latent=exp_latent,
            blend_alpha=self.experience_alpha,
        )


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

        for turn in range(1, args.max_turns + 1):
            step_text, aug_mask = actor.generate(prompt, args.max_new_tokens, args.max_input_tokens)
            action_type, action_text = parse_step_action(step_text)
            event = {
                "turn": turn,
                "step_text": step_text,
                "action_type": action_type,
                "action_text": action_text,
                "aug_mask": aug_mask[:1] if aug_mask is not None else None,
            }
            if getattr(actor, "last_memory_text", None):
                event["memory_text_preview"] = actor.last_memory_text[:1000]
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
            "support_coverage": (len(retrieved_support) / len({normalize_text(t) for t in support})) if support else None,
            "any_support_hit": first_support_hit_turn is not None,
            "first_support_hit_turn": first_support_hit_turn,
            "first_answer_hit_turn": first_answer_hit_turn,
            "all_support_hit_turn": all_support_hit_turn,
            "trajectory": trajectory,
        }
        out_rows.append(summary)
        print(actor.name.upper(), idx, item["dataset"], item.get("id"), "finish", finish_reason, "searches", searches, "cov", summary["support_coverage"], "final_hit", summary["final_answer_hit"])
    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--weaver-model", default=None)
    parser.add_argument("--trigger-model", default=None)
    parser.add_argument("--prompt-latents-len", type=int, default=8)
    parser.add_argument("--inference-latents-len", type=int, default=8)
    parser.add_argument("--max-prompt-aug-num", type=int, default=8)
    parser.add_argument("--max-inference-aug-num", type=int, default=0)
    parser.add_argument("--memgen-load-model-path", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model")
    parser.add_argument("--hotpot-path", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/dev.jsonl")
    parser.add_argument("--twiki-path", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/dev.jsonl")
    parser.add_argument("--oracle-train-path", default="/root/autodl-tmp/memrag_new/multihop_oracle_sft_full_n1000_top3_doc220/train.jsonl")
    parser.add_argument("--num-per-dataset", type=int, default=5)
    parser.add_argument("--experience-pool-size", type=int, default=1000)
    parser.add_argument("--experience-topn", type=int, default=4)
    parser.add_argument("--max-memory-tokens", type=int, default=2048)
    parser.add_argument("--max-state-chars", type=int, default=2200)
    parser.add_argument("--experience-alpha", type=float, default=0.25)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    parser.add_argument("--actors", default="custom,experience", help="Comma-separated: custom,experience")
    parser.add_argument("--output", default="/root/autodl-tmp/memrag_new/memgen_experience_latent_smoke.json")
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
    print("DATA_N", len(data), "EXPERIENCE_N", len(experiences), "ACTORS", args.actors)

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "FlashRAG/Search-R1-compatible multi-turn loop with MemGen-style hidden experience latents",
        "datasets": ["hotpotqa/dev", "2wikimultihopqa/dev"],
        "retriever": "Search-R1 BM25 over wiki-18 via HTTP retrieve API",
        "model": args.model,
        "memgen_load_model_path": args.memgen_load_model_path,
        "settings": vars(args),
        "actors": {},
    }

    actor_names = [x.strip() for x in args.actors.split(",") if x.strip()]
    for actor_name in actor_names:
        if actor_name == "custom":
            actor = MemGenPromptLatentCustomActor(
                args.model,
                args.memgen_load_model_path,
                weaver_model_name=args.weaver_model,
                trigger_model_name=args.trigger_model,
                prompt_latents_len=args.prompt_latents_len,
                inference_latents_len=args.inference_latents_len,
                max_prompt_aug_num=args.max_prompt_aug_num,
                max_inference_aug_num=args.max_inference_aug_num,
            )
        elif actor_name == "experience":
            actor = MemGenExperienceLatentActor(
                args.model,
                args.memgen_load_model_path,
                selector,
                args.max_memory_tokens,
                args.max_state_chars,
                args.experience_alpha,
                weaver_model_name=args.weaver_model,
                trigger_model_name=args.trigger_model,
                prompt_latents_len=args.prompt_latents_len,
                inference_latents_len=args.inference_latents_len,
                max_prompt_aug_num=args.max_prompt_aug_num,
                max_inference_aug_num=args.max_inference_aug_num,
            )
        else:
            raise ValueError(f"unknown actor: {actor_name}")
        print("RUN_ACTOR", actor.name, "LORA_COMPARE_MAX_ABS_DIFF", actor.lora_diff)
        rows = run_actor(actor, data, args)
        result["actors"][actor.name] = {
            "summary": summarize(rows),
            "lora_compare_max_abs_diff": actor.lora_diff,
            "rows": rows,
        }
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
