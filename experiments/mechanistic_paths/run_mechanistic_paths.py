#!/usr/bin/env python3
"""Mechanistic comparison for MemGen-style latent memory paths.

The experiment keeps the visible reasoner prompt fixed and changes only how the
latent memory fed to the reasoner is produced:

1. memgen: trained MemGen weaver on the current prompt only.
2. expel_trained: retrieved ExpeL-like text memory is visible to the trained
   weaver, then converted to latent memory for the reasoner.
3. expel_untrained: the same retrieved text memory is visible to an untrained
   weaver with the same architecture.

For every path the script records next-token/candidate policy metrics, greedy
generation traces, and layer-wise hidden/attention/logit-lens statistics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
MEMGEN_ROOT = REPO_ROOT / "MemGen"
if str(MEMGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(MEMGEN_ROOT))

from memgen.model import MemGenModel  # noqa: E402


ARM_MEMGEN = "memgen"
ARM_EXPEL_TRAINED = "expel_trained"
ARM_EXPEL_UNTRAINED = "expel_untrained"
BASELINE_NO_LATENT = "no_latent"
ARMS = [ARM_MEMGEN, ARM_EXPEL_TRAINED, ARM_EXPEL_UNTRAINED]


@dataclass
class EvalSample:
    sample_id: str
    dataset: str
    question: str
    gold: str
    wrong: str
    aliases: List[str]


@dataclass
class MemoryRecord:
    record_id: str
    question: str
    answer: str
    query: str
    text: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lexical_terms(text: str) -> List[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
        "it", "of", "on", "or", "that", "the", "to", "was", "were", "what", "when", "where", "which",
        "who", "whom", "whose", "why", "with", "did", "does", "do", "this", "these", "those", "there",
    }
    words = re.findall(r"[a-z0-9][a-z0-9_\-']*", normalize_text(text))
    return [w for w in words if len(w) > 2 and w not in stop]


def keyword_query(question: str, limit: int = 8) -> str:
    out = []
    seen = set()
    for term in lexical_terms(question):
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= limit:
            break
    return " ".join(out) if out else question[:120]


def first_answer(row: Dict) -> Tuple[str, List[str]]:
    answer = row.get("answer", {})
    aliases = answer.get("normalized_aliases") or answer.get("aliases") or []
    value = answer.get("normalized_value") or answer.get("value") or (aliases[0] if aliases else "")
    aliases = [str(x) for x in aliases if str(x).strip()]
    if value and value not in aliases:
        aliases.insert(0, str(value))
    return str(value or (aliases[0] if aliases else "")).strip(), aliases


def load_triviaqa(num_samples: int, bank_size: int, seed: int) -> Tuple[List[EvalSample], List[MemoryRecord], List[str]]:
    warnings: List[str] = []
    try:
        from datasets import load_dataset
    except Exception as exc:
        warnings.append(f"datasets import failed: {exc}")
        return fixture_samples(), fixture_bank(), warnings

    try:
        eval_ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split=f"validation[:{num_samples}]")
        train_ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split=f"train[:{bank_size}]")
    except Exception as exc:
        warnings.append(f"triviaqa load failed: {exc}")
        return fixture_samples(), fixture_bank(), warnings

    samples: List[EvalSample] = []
    golds: List[str] = []
    for i, row in enumerate(eval_ds):
        gold, aliases = first_answer(row)
        if not gold:
            continue
        golds.append(gold)
        samples.append(
            EvalSample(
                sample_id=f"triviaqa_val_{i}",
                dataset="triviaqa",
                question=str(row["question"]).strip(),
                gold=gold,
                wrong="",
                aliases=aliases or [gold],
            )
        )
    rng = random.Random(seed)
    for i, sample in enumerate(samples):
        candidates = [g for j, g in enumerate(golds) if j != i and g != sample.gold]
        sample.wrong = rng.choice(candidates) if candidates else "I don't know"

    bank: List[MemoryRecord] = []
    for i, row in enumerate(train_ds):
        answer, aliases = first_answer(row)
        question = str(row["question"]).strip()
        if not question or not answer:
            continue
        query = keyword_query(question)
        text = (
            "Past retrieval experience.\n"
            f"Previous question: {question}\n"
            f"Successful search query: {query}\n"
            f"Observed answer signal: {answer}\n"
            "Policy lesson: preserve named entities, search the bridge entity first, "
            "avoid answering before the retrieved evidence supports the target."
        )
        bank.append(MemoryRecord(record_id=f"triviaqa_train_{i}", question=question, answer=answer, query=query, text=text))
    return samples, bank, warnings


def fixture_samples() -> List[EvalSample]:
    return [
        EvalSample("fixture_0", "fixture", "Who wrote Pride and Prejudice?", "Jane Austen", "Charles Dickens", ["jane austen"]),
        EvalSample("fixture_1", "fixture", "What city contains the Eiffel Tower?", "Paris", "Rome", ["paris"]),
        EvalSample("fixture_2", "fixture", "Which planet is known as the Red Planet?", "Mars", "Venus", ["mars"]),
    ]


def fixture_bank() -> List[MemoryRecord]:
    return [
        MemoryRecord("bank_0", "Who wrote Emma?", "Jane Austen", "Emma Jane Austen author", "Past retrieval experience.\nPrevious question: Who wrote Emma?\nSuccessful search query: Emma Jane Austen author\nObserved answer signal: Jane Austen\nPolicy lesson: search author/title terms before answering."),
        MemoryRecord("bank_1", "Where is the Louvre Museum?", "Paris", "Louvre Museum Paris", "Past retrieval experience.\nPrevious question: Where is the Louvre Museum?\nSuccessful search query: Louvre Museum Paris\nObserved answer signal: Paris\nPolicy lesson: search entity plus location cue."),
        MemoryRecord("bank_2", "What planet has Olympus Mons?", "Mars", "Olympus Mons planet Mars", "Past retrieval experience.\nPrevious question: What planet has Olympus Mons?\nSuccessful search query: Olympus Mons planet Mars\nObserved answer signal: Mars\nPolicy lesson: search the distinctive entity first."),
    ]


class BM25MemoryIndex:
    def __init__(self, records: List[MemoryRecord], k1: float = 1.5, b: float = 0.75):
        self.records = records
        self.k1 = k1
        self.b = b
        self.docs = [lexical_terms(r.question + " " + r.query + " " + r.answer) for r in records]
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        df = Counter()
        for terms in self.docs:
            df.update(set(terms))
        n = max(1, len(self.docs))
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.tfs = [Counter(terms) for terms in self.docs]

    def search(self, query: str, topk: int) -> List[Tuple[MemoryRecord, float]]:
        terms = lexical_terms(query)
        scored = []
        for rec, tf, dl in zip(self.records, self.tfs, self.doc_lens):
            score = 0.0
            for term in terms:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            scored.append((rec, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:topk]


def build_prompt(tokenizer, question: str) -> str:
    content = (
        "You are a retrieval QA agent. Choose exactly one next action.\n"
        "Valid actions:\n"
        "<search> concise wikipedia search query </search>\n"
        "<answer> short answer </answer>\n\n"
        f"Question: {question}\n"
        "Next action:"
    )
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return content


def build_weaver_context(prompt: str, retrieved_memory: Optional[str]) -> str:
    if not retrieved_memory:
        return prompt
    return (
        f"{prompt}\n\n"
        "Retrieved experience memory for the latent weaver only:\n"
        f"{retrieved_memory}\n"
        "Use the retrieved experience as policy memory, not as direct evidence."
    )


def derive_retrieval_query(tokenizer, prompt: str, max_tokens: int = 96) -> str:
    """Decode the current state into the query used by the external memory DB.

    Appendix E describes q_{t,j} as a natural-language query decoded from the
    generated/current token sequence.  In this prompt-end experiment, the current
    sequence is the user prompt.  We decode it and keep the task-bearing part so
    retrieval is driven by state text rather than the raw sample object.
    """

    ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    if "Question:" in decoded:
        decoded = decoded.split("Question:", 1)[-1]
    if "Next action:" in decoded:
        decoded = decoded.split("Next action:", 1)[0]
    query = keyword_query(decoded, limit=10)
    return query or normalize_text(decoded)[:200] or decoded[:200]


def make_candidates(sample: EvalSample, retrieved_records: List[MemoryRecord]) -> List[Dict[str, str]]:
    query = keyword_query(sample.question)
    memory_query = retrieved_records[0].query if retrieved_records else query
    return [
        {"id": "search_current", "text": f"<search> {query} </search>"},
        {"id": "search_memory", "text": f"<search> {memory_query} </search>"},
        {"id": "gold_answer", "text": f"<answer> {sample.gold} </answer>"},
        {"id": "distractor_answer", "text": f"<answer> {sample.wrong} </answer>"},
        {"id": "malformed", "text": "I should think more before choosing a valid action."},
    ]


def build_memgen_config(args, load_path: Optional[str]) -> Dict:
    lora = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.lora_target_modules.split(","),
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    return {
        "model_name": args.model,
        "load_model_path": load_path,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "max_prompt_aug_num": 1,
        "max_inference_aug_num": 0,
        "weaver": {
            "model_name": args.weaver_model or args.model,
            "prompt_latents_len": args.latent_len,
            "inference_latents_len": args.latent_len,
            "lora_config": lora,
        },
        "trigger": {
            "model_name": args.trigger_model or args.model,
            "active": False,
            "lora_config": lora,
        },
    }


def load_memgen(args, load_path: Optional[str]) -> MemGenModel:
    model = MemGenModel.from_config(build_memgen_config(args, load_path)).eval()
    if args.torch_dtype != "float32":
        model = model.to(getattr(torch, args.torch_dtype))
    if torch.cuda.is_available() and args.device == "cuda":
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    if model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.cumsum(-1) - 1).clamp(min=0).long()


def inject_latents(reasoner, tokenizer, prompt: str, latents: Optional[torch.Tensor], max_prompt_tokens: int):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_prompt_tokens).to(reasoner.device)
    prompt_embeds = reasoner.get_input_embeddings()(encoded["input_ids"])
    if latents is None:
        attention_mask = encoded["attention_mask"]
        return prompt_embeds, attention_mask, make_position_ids(attention_mask), torch.zeros_like(attention_mask, dtype=torch.bool)
    latents = latents.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    latent_mask = torch.ones(latents.shape[:-1], dtype=encoded["attention_mask"].dtype, device=prompt_embeds.device)
    inputs_embeds = torch.cat([prompt_embeds, latents], dim=1)
    attention_mask = torch.cat([encoded["attention_mask"], latent_mask], dim=1)
    position_ids = make_position_ids(attention_mask)
    latent_positions = torch.cat(
        [
            torch.zeros(encoded["attention_mask"].shape, dtype=torch.bool, device=prompt_embeds.device),
            torch.ones(latent_mask.shape, dtype=torch.bool, device=prompt_embeds.device),
        ],
        dim=1,
    )
    return inputs_embeds, attention_mask, position_ids, latent_positions


def compute_latents(memgen: MemGenModel, tokenizer, weaver_context: str, max_context_tokens: int) -> torch.Tensor:
    reasoner = memgen.reasoner
    encoded = tokenizer(
        weaver_context,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_context_tokens,
    ).to(memgen.device)
    with torch.no_grad():
        reasoner_embeds = reasoner.get_input_embeddings()(encoded["input_ids"])
        position_ids = make_position_ids(encoded["attention_mask"])
        weaver_inputs = memgen.reasoner_to_weaver(reasoner_embeds)
        weaver_hidden, _, _ = memgen.weaver.augment_prompt(weaver_inputs, encoded["attention_mask"], position_ids)
        latents = memgen.weaver_to_reasoner(weaver_hidden)
    return latents.detach()


def compute_appendix_e_latents(
    memgen: MemGenModel,
    tokenizer,
    prompt: str,
    retrieved_memory: Optional[str],
    max_context_tokens: int,
    max_retrieval_memory_tokens: int,
) -> torch.Tensor:
    """Generate MemGen latents with Appendix-E-style retrieval fusion.

    The current state is encoded as reasoner embeddings and projected into the
    weaver space. Retrieved textual memory snippets are encoded separately,
    projected into the same weaver space, concatenated with the current state,
    and then consumed by the trained/untrained weaver to synthesize latent memory.
    The reasoner never sees the retrieved text directly.
    """

    reasoner = memgen.reasoner
    state = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_context_tokens,
    ).to(memgen.device)
    with torch.no_grad():
        state_embeds = reasoner.get_input_embeddings()(state["input_ids"])
        weaver_inputs = memgen.reasoner_to_weaver(state_embeds)
        attention_mask = state["attention_mask"]

        if retrieved_memory:
            memory = tokenizer(
                retrieved_memory,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_retrieval_memory_tokens,
            ).to(memgen.device)
            memory_embeds = reasoner.get_input_embeddings()(memory["input_ids"])
            memory_weaver_inputs = memgen.reasoner_to_weaver(memory_embeds)
            weaver_inputs = torch.cat([weaver_inputs, memory_weaver_inputs], dim=1)
            attention_mask = torch.cat([attention_mask, memory["attention_mask"]], dim=1)

        position_ids = make_position_ids(attention_mask)
        weaver_hidden, _, _ = memgen.weaver.augment_prompt(weaver_inputs, attention_mask, position_ids)
        latents = memgen.weaver_to_reasoner(weaver_hidden)
    return latents.detach()


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> List[Dict]:
    probs = F.softmax(logits.float(), dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [{"id": int(idx), "text": tokenizer.decode([idx]), "prob": float(prob)} for prob, idx in zip(vals.tolist(), idxs.tolist())]


def entropy_from_logits(logits: torch.Tensor) -> float:
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return float(-(p * logp).sum().item())


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p_log = F.log_softmax(p_logits.float(), dim=-1)
    q_log = F.log_softmax(q_logits.float(), dim=-1)
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum().item())


def candidate_policy(scores: Dict[str, float]) -> Dict[str, float]:
    keys = list(scores)
    vals = torch.tensor([scores[k] for k in keys], dtype=torch.float32)
    probs = F.softmax(vals, dim=0).tolist()
    return {k: float(v) for k, v in zip(keys, probs)}


def score_candidate(reasoner, tokenizer, prompt: str, latents: Optional[torch.Tensor], candidate: str, max_prompt_tokens: int) -> Tuple[float, float, int]:
    candidate_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False)["input_ids"].to(reasoner.device)
    inputs_embeds, attention_mask, position_ids, _ = inject_latents(reasoner, tokenizer, prompt, latents, max_prompt_tokens)
    cand_embeds = reasoner.get_input_embeddings()(candidate_ids)
    start = inputs_embeds.size(1)
    inputs_embeds = torch.cat([inputs_embeds, cand_embeds], dim=1)
    cand_mask = torch.ones(candidate_ids.shape, dtype=attention_mask.dtype, device=reasoner.device)
    attention_mask = torch.cat([attention_mask, cand_mask], dim=1)
    position_ids = make_position_ids(attention_mask)
    with torch.no_grad():
        outputs = reasoner(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    logits = outputs.logits[0]
    logps = []
    for j, token_id in enumerate(candidate_ids[0]):
        pos = start + j - 1
        logps.append(F.log_softmax(logits[pos].float(), dim=-1)[token_id])
    stacked = torch.stack(logps)
    return float(stacked.sum().item()), float(stacked.mean().item()), int(candidate_ids.size(1))


def model_norm(reasoner, hidden: torch.Tensor) -> torch.Tensor:
    base = getattr(reasoner, "model", None)
    norm = getattr(base, "norm", None)
    if norm is None:
        return hidden
    return norm(hidden)


def first_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(ids[0])


def layer_probe_after_suffix(
    reasoner,
    tokenizer,
    prompt: str,
    latents: Optional[torch.Tensor],
    suffix: str,
    positive_text: str,
    negative_text: str,
    positive_name: str,
    negative_name: str,
    max_prompt_tokens: int,
) -> List[Dict]:
    suffix_ids = tokenizer(suffix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(reasoner.device)
    positive_id = first_token_id(tokenizer, positive_text)
    negative_id = first_token_id(tokenizer, negative_text)
    inputs_embeds, attention_mask, _, _ = inject_latents(reasoner, tokenizer, prompt, latents, max_prompt_tokens)
    suffix_embeds = reasoner.get_input_embeddings()(suffix_ids)
    inputs_embeds = torch.cat([inputs_embeds, suffix_embeds], dim=1)
    suffix_mask = torch.ones(suffix_ids.shape, dtype=attention_mask.dtype, device=reasoner.device)
    attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)
    position_ids = make_position_ids(attention_mask)
    with torch.no_grad():
        outputs = reasoner(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    rows = []
    for layer_idx, hidden in enumerate(outputs.hidden_states):
        last_hidden = hidden[:, -1:, :]
        with torch.no_grad():
            lens_logits = reasoner.lm_head(model_norm(reasoner, last_hidden))[0, -1].detach().float().cpu()
        probs = F.softmax(lens_logits, dim=-1)
        rows.append(
            {
                "layer": layer_idx,
                f"{positive_name}_token_id": int(positive_id),
                f"{negative_name}_token_id": int(negative_id),
                f"{positive_name}_prob": float(probs[positive_id].item()),
                f"{negative_name}_prob": float(probs[negative_id].item()),
                f"{positive_name}_vs_{negative_name}_logit_margin": float(lens_logits[positive_id].item() - lens_logits[negative_id].item()),
            }
        )
    del outputs
    return rows


def forward_probe(
    reasoner,
    tokenizer,
    prompt: str,
    latents: Optional[torch.Tensor],
    candidates: List[Dict[str, str]],
    gold_text: str,
    wrong_text: str,
    max_prompt_tokens: int,
    top_k: int,
    collect_attentions: bool,
) -> Tuple[Dict, torch.Tensor, List[torch.Tensor]]:
    inputs_embeds, attention_mask, position_ids, latent_positions = inject_latents(reasoner, tokenizer, prompt, latents, max_prompt_tokens)
    with torch.no_grad():
        outputs = reasoner(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=collect_attentions,
        )
    next_logits = outputs.logits[0, -1].detach().float().cpu()
    hidden_states = [h[0, -1].detach().float().cpu() for h in outputs.hidden_states]
    candidate_totals, candidate_avgs, candidate_lens = {}, {}, {}
    for cand in candidates:
        total, avg, length = score_candidate(reasoner, tokenizer, prompt, latents, cand["text"], max_prompt_tokens)
        candidate_totals[cand["id"]] = total
        candidate_avgs[cand["id"]] = avg
        candidate_lens[cand["id"]] = length
    policy = candidate_policy(candidate_avgs)

    layer_action_key_lens = layer_probe_after_suffix(
        reasoner,
        tokenizer,
        prompt,
        latents,
        "<",
        "search",
        "answer",
        "search",
        "answer",
        max_prompt_tokens,
    )
    layer_answer_content_lens = layer_probe_after_suffix(
        reasoner,
        tokenizer,
        prompt,
        latents,
        "<answer> ",
        gold_text,
        wrong_text,
        "gold",
        "distractor",
        max_prompt_tokens,
    )

    attention_mass = []
    if collect_attentions and getattr(outputs, "attentions", None) is not None:
        latent_bool = latent_positions[0].bool()
        if latent_bool.any():
            for layer_idx, attn in enumerate(outputs.attentions):
                # [B, heads, query, key]; measure final decision state's mass on latent slots.
                layer_attn = attn[0, :, -1, :].detach().float().cpu()
                mass = layer_attn[:, latent_bool.detach().cpu()].sum(dim=-1)
                attention_mass.append({"layer": layer_idx + 1, "latent_attention_mass": float(mass.mean().item())})

    row = {
        "sequence_len": int(attention_mask.size(1)),
        "latent_count": int(latents.size(1)) if latents is not None else 0,
        "next_token_entropy": entropy_from_logits(next_logits),
        "top_tokens": top_tokens(tokenizer, next_logits, top_k),
        "candidate_total_logprobs": candidate_totals,
        "candidate_avg_logprobs": candidate_avgs,
        "candidate_lengths": candidate_lens,
        "candidate_policy": policy,
        "search_prob": float(policy.get("search_current", 0.0) + policy.get("search_memory", 0.0)),
        "answer_prob": float(policy.get("gold_answer", 0.0) + policy.get("distractor_answer", 0.0)),
        "gold_answer_prob": float(policy.get("gold_answer", 0.0)),
        "search_vs_answer_margin": float(max(candidate_avgs["search_current"], candidate_avgs["search_memory"]) - candidate_avgs["gold_answer"]),
        "gold_vs_distractor_margin": float(candidate_avgs["gold_answer"] - candidate_avgs["distractor_answer"]),
        "layer_action_key_lens": layer_action_key_lens,
        "layer_answer_content_lens": layer_answer_content_lens,
        "latent_attention_mass": attention_mass,
    }
    del outputs
    return row, next_logits, hidden_states


def greedy_generate(
    reasoner,
    tokenizer,
    prompt: str,
    latents: Optional[torch.Tensor],
    max_prompt_tokens: int,
    max_new_tokens: int,
    top_k: int,
) -> Dict:
    inputs_embeds, attention_mask, _, _ = inject_latents(reasoner, tokenizer, prompt, latents, max_prompt_tokens)
    generated_ids: List[int] = []
    trace = []
    for step in range(max_new_tokens):
        position_ids = make_position_ids(attention_mask)
        with torch.no_grad():
            outputs = reasoner(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        logits = outputs.logits[0, -1].detach().float().cpu()
        token_id = int(torch.argmax(logits).item())
        generated_ids.append(token_id)
        if step < 8:
            trace.append(
                {
                    "step": step,
                    "chosen_id": token_id,
                    "chosen_text": tokenizer.decode([token_id]),
                    "entropy": entropy_from_logits(logits),
                    "top_tokens": top_tokens(tokenizer, logits, top_k),
                }
            )
        if token_id == tokenizer.eos_token_id:
            break
        next_id = torch.tensor([[token_id]], dtype=torch.long, device=reasoner.device)
        next_embed = reasoner.get_input_embeddings()(next_id)
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)
        next_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)
        del outputs
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {"text": text, "token_ids": generated_ids, "trace": trace}


def add_hidden_drift(rows_by_arm: Dict[str, Dict], hidden_by_arm: Dict[str, List[torch.Tensor]], logits_by_arm: Dict[str, torch.Tensor]) -> Dict:
    comparisons: Dict[str, Dict] = {}
    if BASELINE_NO_LATENT in hidden_by_arm:
        ref_hidden = hidden_by_arm[BASELINE_NO_LATENT]
        ref_logits = logits_by_arm[BASELINE_NO_LATENT]
        for arm in ARMS:
            hiddens = hidden_by_arm[arm]
            layer_rows = []
            for idx, (h, ref) in enumerate(zip(hiddens, ref_hidden)):
                layer_rows.append(
                    {
                        "layer": idx,
                        "cosine_to_no_latent": float(F.cosine_similarity(h, ref, dim=0).item()),
                        "l2_to_no_latent": float(torch.norm(h - ref).item()),
                    }
                )
            rows_by_arm[arm]["hidden_drift_to_no_latent"] = layer_rows
            comparisons[f"{BASELINE_NO_LATENT}_to_{arm}"] = {"next_token_kl": kl_from_logits(ref_logits, logits_by_arm[arm])}
    for left, right in [(ARM_MEMGEN, ARM_EXPEL_TRAINED), (ARM_EXPEL_TRAINED, ARM_EXPEL_UNTRAINED), (ARM_MEMGEN, ARM_EXPEL_UNTRAINED)]:
        if left not in hidden_by_arm or right not in hidden_by_arm:
            continue
        layer_rows = []
        for idx, (h_l, h_r) in enumerate(zip(hidden_by_arm[left], hidden_by_arm[right])):
            layer_rows.append(
                {
                    "layer": idx,
                    "cosine": float(F.cosine_similarity(h_l, h_r, dim=0).item()),
                    "l2": float(torch.norm(h_l - h_r).item()),
                }
            )
        comparisons[f"{left}_to_{right}"] = {
            "next_token_kl": kl_from_logits(logits_by_arm[left], logits_by_arm[right]),
            "layer_hidden_distance": layer_rows,
        }
    return comparisons


def run_trained_phase(args, samples: List[EvalSample], index: BM25MemoryIndex) -> Tuple[List[Dict], Dict]:
    model = load_memgen(args, args.memgen_load_model_path)
    tokenizer = model.tokenizer
    rows = []
    for sample_idx, sample in enumerate(samples):
        prompt = build_prompt(tokenizer, sample.question)
        retrieval_query = derive_retrieval_query(tokenizer, prompt, args.retrieval_query_tokens)
        retrieved = index.search(retrieval_query, args.memory_topk)
        retrieved_records = [rec for rec, _ in retrieved]
        retrieved_memory = "\n\n".join(rec.text for rec in retrieved_records)
        candidates = make_candidates(sample, retrieved_records)
        if args.retrieval_fusion == "appendix_e_embedding":
            latents = {
                ARM_MEMGEN: compute_appendix_e_latents(
                    model,
                    tokenizer,
                    prompt,
                    None,
                    args.max_weaver_context_tokens,
                    args.max_retrieval_memory_tokens,
                ),
                ARM_EXPEL_TRAINED: compute_appendix_e_latents(
                    model,
                    tokenizer,
                    prompt,
                    retrieved_memory,
                    args.max_weaver_context_tokens,
                    args.max_retrieval_memory_tokens,
                ),
            }
        else:
            contexts = {
                ARM_MEMGEN: build_weaver_context(prompt, None),
                ARM_EXPEL_TRAINED: build_weaver_context(prompt, retrieved_memory),
            }
            latents = {
                ARM_MEMGEN: compute_latents(model, tokenizer, contexts[ARM_MEMGEN], args.max_weaver_context_tokens),
                ARM_EXPEL_TRAINED: compute_latents(model, tokenizer, contexts[ARM_EXPEL_TRAINED], args.max_weaver_context_tokens),
            }
        arms: Dict[str, Dict] = {}
        logits: Dict[str, torch.Tensor] = {}
        hiddens: Dict[str, List[torch.Tensor]] = {}
        base_row, base_logits, base_hiddens = forward_probe(
            model.reasoner,
            tokenizer,
            prompt,
            None,
            candidates,
            sample.gold,
            sample.wrong,
            args.max_prompt_tokens,
            args.top_tokens,
            False,
        )
        arms[BASELINE_NO_LATENT] = base_row
        logits[BASELINE_NO_LATENT] = base_logits
        hiddens[BASELINE_NO_LATENT] = base_hiddens
        for arm in [ARM_MEMGEN, ARM_EXPEL_TRAINED]:
            arm_row, arm_logits, arm_hiddens = forward_probe(
                model.reasoner,
                tokenizer,
                prompt,
                latents[arm],
                candidates,
                sample.gold,
                sample.wrong,
                args.max_prompt_tokens,
                args.top_tokens,
                args.collect_attentions,
            )
            arm_row["generation"] = greedy_generate(
                model.reasoner,
                tokenizer,
                prompt,
                latents[arm],
                args.max_prompt_tokens,
                args.max_new_tokens,
                args.top_tokens,
            )
            arm_row["latent_norm_mean"] = float(latents[arm].float().norm(dim=-1).mean().item())
            arm_row["latent_norm_std"] = float(latents[arm].float().norm(dim=-1).std().item())
            arms[arm] = arm_row
            logits[arm] = arm_logits
            hiddens[arm] = arm_hiddens

        row = {
            "idx": sample_idx,
            "id": sample.sample_id,
            "dataset": sample.dataset,
            "question": sample.question,
            "gold": sample.gold,
            "wrong": sample.wrong,
            "aliases": sample.aliases,
            "retrieval_fusion": args.retrieval_fusion,
            "retrieval_query": retrieval_query,
            "retrieved_memory": [
                {"record_id": rec.record_id, "score": float(score), "question": rec.question, "answer": rec.answer, "query": rec.query}
                for rec, score in retrieved
            ],
            "prompt_preview": prompt[-500:],
            "candidates": candidates,
            "arms": arms,
            "_logits": logits,
            "_hiddens": hiddens,
        }
        rows.append(row)
        print("TRAINED_ROW", sample_idx, sample.sample_id, "memgen_gold_prob", arms[ARM_MEMGEN]["gold_answer_prob"], "expel_trained_gold_prob", arms[ARM_EXPEL_TRAINED]["gold_answer_prob"], flush=True)
        torch.cuda.empty_cache()
    metadata = {"tokenizer_name": args.model}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows, metadata


def run_untrained_phase(args, rows: List[Dict]) -> None:
    torch.manual_seed(args.untrained_seed)
    model = load_memgen(args, None)
    tokenizer = model.tokenizer
    for row in rows:
        sample = EvalSample(row["id"], row["dataset"], row["question"], row["gold"], row["wrong"], row["aliases"])
        prompt = build_prompt(tokenizer, sample.question)
        retrieved_texts = []
        for rec in row["retrieved_memory"]:
            retrieved_texts.append(
                "Past retrieval experience.\n"
                f"Previous question: {rec['question']}\n"
                f"Successful search query: {rec['query']}\n"
                f"Observed answer signal: {rec['answer']}\n"
                "Policy lesson: preserve named entities, search the bridge entity first, "
                "avoid answering before the retrieved evidence supports the target."
            )
        retrieved_memory = "\n\n".join(retrieved_texts)
        candidates = row["candidates"]
        if args.retrieval_fusion == "appendix_e_embedding":
            latent = compute_appendix_e_latents(
                model,
                tokenizer,
                prompt,
                retrieved_memory,
                args.max_weaver_context_tokens,
                args.max_retrieval_memory_tokens,
            )
        else:
            context = build_weaver_context(prompt, retrieved_memory)
            latent = compute_latents(model, tokenizer, context, args.max_weaver_context_tokens)
        arm_row, arm_logits, arm_hiddens = forward_probe(
            model.reasoner,
            tokenizer,
            prompt,
            latent,
            candidates,
            sample.gold,
            sample.wrong,
            args.max_prompt_tokens,
            args.top_tokens,
            args.collect_attentions,
        )
        arm_row["generation"] = greedy_generate(
            model.reasoner,
            tokenizer,
            prompt,
            latent,
            args.max_prompt_tokens,
            args.max_new_tokens,
            args.top_tokens,
        )
        arm_row["latent_norm_mean"] = float(latent.float().norm(dim=-1).mean().item())
        arm_row["latent_norm_std"] = float(latent.float().norm(dim=-1).std().item())
        row["arms"][ARM_EXPEL_UNTRAINED] = arm_row
        row["_logits"][ARM_EXPEL_UNTRAINED] = arm_logits
        row["_hiddens"][ARM_EXPEL_UNTRAINED] = arm_hiddens
        row["comparisons"] = add_hidden_drift(row["arms"], row["_hiddens"], row["_logits"])
        print("UNTRAINED_ROW", row["idx"], row["id"], "gold_prob", arm_row["gold_answer_prob"], flush=True)
        torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def strip_tensors(row: Dict) -> Dict:
    row = dict(row)
    row.pop("_logits", None)
    row.pop("_hiddens", None)
    return row


def summarize_rows(rows: List[Dict]) -> Dict:
    summary: Dict[str, Dict] = {"n": len(rows), "arms": {}}
    for arm in [BASELINE_NO_LATENT] + ARMS:
        arm_rows = [r["arms"][arm] for r in rows if arm in r["arms"]]
        if not arm_rows:
            continue
        summary["arms"][arm] = {
            "gold_answer_prob_mean": sum(r["gold_answer_prob"] for r in arm_rows) / len(arm_rows),
            "search_prob_mean": sum(r["search_prob"] for r in arm_rows) / len(arm_rows),
            "answer_prob_mean": sum(r["answer_prob"] for r in arm_rows) / len(arm_rows),
            "gold_vs_distractor_margin_mean": sum(r["gold_vs_distractor_margin"] for r in arm_rows) / len(arm_rows),
            "search_vs_answer_margin_mean": sum(r["search_vs_answer_margin"] for r in arm_rows) / len(arm_rows),
            "next_token_entropy_mean": sum(r["next_token_entropy"] for r in arm_rows) / len(arm_rows),
            "latent_norm_mean": sum(r.get("latent_norm_mean", 0.0) for r in arm_rows) / len(arm_rows),
        }
    comp_keys = sorted({key for row in rows for key in row.get("comparisons", {}) if "next_token_kl" in row["comparisons"][key]})
    summary["comparisons"] = {}
    for key in comp_keys:
        vals = [row["comparisons"][key]["next_token_kl"] for row in rows if key in row.get("comparisons", {})]
        summary["comparisons"][key] = {"next_token_kl_mean": sum(vals) / len(vals), "n": len(vals)}
    return summary


def run(args) -> Dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
    print("MODEL", args.model)
    print("CHECKPOINT", args.memgen_load_model_path)
    samples, bank, warnings = load_triviaqa(args.num_samples, args.memory_bank_size, args.seed)
    index = BM25MemoryIndex(bank)
    rows, metadata = run_trained_phase(args, samples, index)
    run_untrained_phase(args, rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(strip_tensors(row), ensure_ascii=False) + "\n")
    summary = summarize_rows(rows)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schema": "one JSON object per eval sample; all tensor fields stripped",
        "experiment": "MemGen vs retrieval-conditioned trained weaver vs retrieval-conditioned untrained weaver",
        "settings": vars(args),
        "warnings": warnings,
        "summary": summary,
        **metadata,
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    print("RESULT_PATH", output_path)
    print("META_PATH", meta_path)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--weaver-model", default=None)
    parser.add_argument("--trigger-model", default=None)
    parser.add_argument("--memgen-load-model-path", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--memory-bank-size", type=int, default=800)
    parser.add_argument("--memory-topk", type=int, default=3)
    parser.add_argument("--retrieval-fusion", choices=["text_context", "appendix_e_embedding"], default="text_context")
    parser.add_argument("--retrieval-query-tokens", type=int, default=128)
    parser.add_argument("--latent-len", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--max-weaver-context-tokens", type=int, default=1024)
    parser.add_argument("--max-retrieval-memory-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--top-tokens", type=int, default=8)
    parser.add_argument("--collect-attentions", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--untrained-seed", type=int, default=1234)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", default="results/mechanistic_paths/qwen15_memgen_expel_paths_n30.jsonl")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
