#!/usr/bin/env python3
"""Memory carriers for the policy-fidelity playground."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "with",
}


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def keywords(text: str, limit: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']*", text.lower())
    out = []
    seen = set()
    for word in words:
        if word in STOPWORDS or len(word) <= 2 or word in seen:
            continue
        seen.add(word)
        out.append(word)
        if len(out) >= limit:
            break
    return out


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()


def build_full_memory(sample: Dict) -> str:
    """Create the original memory m used by every carrier.

    This is a controlled playground memory, not a retrieved evidence passage.
    It deliberately contains both policy hints and task outcome information so
    that policy preservation can be measured against a strong full-memory arm.
    """

    question = sample["question"]
    gold = sample["gold"]
    key_terms = ", ".join(keywords(question, limit=6)) or "question terms"
    return normalize_space(
        f"""
        Past successful memory record.
        Task family: {sample.get('dataset', 'qa')}.
        Current-style question: {question}
        Useful policy: identify the requested target, avoid unsupported guesses,
        and prefer the answer candidate that is explicitly supported by this memory.
        Key terms to preserve: {key_terms}.
        Correct answer signal: {gold}.
        Common failure pattern: choosing a fluent but unsupported distractor answer
        or stopping before the target entity/value is checked.
        Compression note: a useful compressed memory should preserve the answer
        signal and the decision policy, not necessarily reconstruct this text.
        """
    )


def extractive_memory(tokenizer, memory: str, max_tokens: int) -> str:
    priority = []
    for sent in re.split(r"(?<=[.!?])\s+", memory):
        low = sent.lower()
        if "correct answer" in low or "useful policy" in low or "key terms" in low:
            priority.append(sent)
    text = " ".join(priority) if priority else memory
    return truncate_to_tokens(tokenizer, normalize_space(text), max_tokens)


class SummaryCache:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.data: Dict[str, str] = {}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")


def generated_summary_memory(
    model,
    tokenizer,
    memory: str,
    max_tokens: int,
    cache: Optional[SummaryCache] = None,
    max_new_tokens: int = 48,
) -> str:
    key = stable_hash(f"{max_tokens}\n{memory}")
    cached = cache.get(key) if cache else None
    if cached is not None:
        return cached

    prompt = (
        "Compress the memory into a short policy note. Preserve the correct answer signal "
        "and the decision policy. Do not add new facts.\n\n"
        f"Memory:\n{memory}\n\nCompressed memory:"
    )
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    summary = truncate_to_tokens(tokenizer, normalize_space(decoded), max_tokens)
    if not summary:
        summary = extractive_memory(tokenizer, memory, max_tokens)
    if cache:
        cache.set(key, summary)
    return summary


def mean_pool_latents(model, tokenizer, memory: str, latent_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = tokenizer(memory, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    if ids.numel() == 0:
        hidden = model.get_input_embeddings().weight.shape[1]
        return torch.zeros(1, latent_len, hidden, device=device, dtype=dtype)
    embeds = model.get_input_embeddings()(ids).to(dtype=dtype)
    chunks = torch.tensor_split(embeds[0], latent_len, dim=0)
    pooled = []
    for chunk in chunks:
        if chunk.numel() == 0:
            pooled.append(torch.zeros(embeds.size(-1), device=device, dtype=dtype))
        else:
            pooled.append(chunk.mean(dim=0))
    return torch.stack(pooled, dim=0).unsqueeze(0)


def random_latents_like(latents: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=latents.device)
    generator.manual_seed(int(seed))
    random = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    target_norm = latents.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    random_norm = random.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return random * (target_norm / random_norm)


def fixed_soft_prompt_latents(memgen_model, latent_len: int, dtype: torch.dtype) -> torch.Tensor:
    latents = memgen_model.weaver.prompt_query_latents[:latent_len].detach().unsqueeze(0)
    if latents.size(1) < latent_len:
        repeat = latent_len - latents.size(1)
        pad = latents[:, -1:, :].repeat(1, repeat, 1)
        latents = torch.cat([latents, pad], dim=1)
    if hasattr(memgen_model.weaver, "prompt_latent_ln"):
        latents = memgen_model.weaver.prompt_latent_ln(latents)
    if hasattr(memgen_model.weaver, "prompt_latent_scale"):
        latents = latents * memgen_model.weaver.prompt_latent_scale.detach()
    latents = memgen_model.weaver_to_reasoner(latents)
    return latents.to(device=memgen_model.device, dtype=dtype)


def make_action_candidates(sample: Dict, wrong_answer: str) -> List[Dict[str, str]]:
    question = sample["question"]
    gold = sample["gold"]
    query = " ".join(keywords(question, limit=7)) or question[:80]
    return [
        {"id": "gold", "text": f"Answer: {gold}"},
        {"id": "distractor", "text": f"Answer: {wrong_answer}"},
        {"id": "abstain", "text": "Answer: I don't know."},
        {"id": "search", "text": f"Search: {query}"},
        {"id": "malformed", "text": "I should think about this but will not choose a valid action."},
    ]


def batched(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
