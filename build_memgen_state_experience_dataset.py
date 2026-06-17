#!/usr/bin/env python3
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from flashrag.pipeline.reasoning_pipeline import SearchR1Pipeline
    USER_PROMPT = SearchR1Pipeline.user_prompt
except Exception:
    USER_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}"""

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "had",
    "has", "have", "he", "her", "his", "in", "is", "it", "its", "of", "on", "or", "she", "that",
    "the", "their", "this", "to", "was", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "with", "you", "your", "question", "answer", "film", "name",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def iter_jsonl(path: str, limit: int = 0) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def build_initial_prompt(question: str) -> str:
    user = USER_PROMPT.format(question=question)
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def extract_searches(messages: List[Dict]) -> List[str]:
    out = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for m in re.finditer(r"<search>(.*?)</search>", msg.get("content", ""), flags=re.S | re.I):
            q = re.sub(r"\s+", " ", m.group(1)).strip()
            if q:
                out.append(q)
    return out


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


def pool_item(row: Dict) -> Dict:
    question = row.get("question", "")
    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "question": question,
        "terms": set(normalize_text(question).split()) - STOP_WORDS,
        "supporting_titles": row.get("supporting_titles", []),
        "golden_answers": row.get("golden_answers", []),
        "searches": extract_searches(row.get("messages", []))[:4],
        "doc_titles": extract_doc_titles(row.get("messages", []), max_titles=6),
    }


def select_experiences(row: Dict, pool: List[Dict], topn: int) -> List[Dict]:
    q_terms = set(normalize_text(row.get("question", "")).split()) - STOP_WORDS
    scored = []
    row_id = row.get("id")
    for ex in pool:
        if ex.get("id") == row_id:
            continue
        overlap = len(q_terms & ex["terms"])
        dataset_bonus = 1 if ex.get("dataset") == row.get("dataset") else 0
        if overlap or dataset_bonus:
            scored.append((overlap * 2 + dataset_bonus, ex))
    if not scored:
        scored = [(0, ex) for ex in pool if ex.get("id") != row_id]
    scored.sort(key=lambda x: (-x[0], x[1].get("id") or ""))
    return [ex for _, ex in scored[:topn]]


def format_experience_text(row: Dict, prompt: str, examples: List[Dict], max_state_chars: int) -> str:
    blocks = [
        "Retrieval experience memory for Search-R1 style multi-hop QA.",
        "The hidden memory should guide only the next action policy: generate valid <think>...</think> plus either <search>...</search> or <answer>...</answer>. Prefer searches that hit new supporting facts, avoid duplicate/no-hit searches, and answer only after enough evidence is in the current state.",
    ]
    for i, ex in enumerate(examples, 1):
        searches = "; ".join(f"S{j + 1}: {q}" for j, q in enumerate(ex.get("searches", [])))
        supports = "; ".join(ex.get("supporting_titles", [])[:4])
        docs = "; ".join(ex.get("doc_titles", [])[:5])
        blocks.append(
            f"Experience {i}:\n"
            f"Question: {ex.get('question', '')}\n"
            f"Useful searches: {searches}\n"
            f"Supporting fact targets: {supports}\n"
            f"Retrieved evidence titles: {docs}"
        )
    blocks.append(
        "Current state to guide, without revealing the next action label:\n"
        f"Question: {row.get('question', '')}\n"
        f"State tail:\n{prompt[-max_state_chars:]}"
    )
    return "\n\n".join(blocks)


def action_type(text: str) -> str:
    if re.search(r"<search>.*?</search>", text, flags=re.S | re.I):
        return "search"
    if re.search(r"<answer>.*?</answer>", text, flags=re.S | re.I):
        return "answer"
    return "other"


def make_state_samples(row: Dict, pool: List[Dict], args) -> List[Dict]:
    messages = row.get("messages", [])
    if not messages:
        return []
    prompt = build_initial_prompt(row.get("question", ""))
    samples = []
    turn = 0
    examples = select_experiences(row, pool, args.experience_topn)
    for idx, msg in enumerate(messages[1:], 1):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            turn += 1
            typ = action_type(content)
            if typ in {"search", "answer"}:
                samples.append({
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "question": row.get("question", ""),
                    "turn": turn,
                    "action_type": typ,
                    "prompt": prompt,
                    "experience_text": format_experience_text(row, prompt, examples, args.max_state_chars),
                    "target_text": content.strip(),
                    "supporting_titles": row.get("supporting_titles", []),
                    "golden_answers": row.get("golden_answers", []),
                    "experience_ids": [ex.get("id") for ex in examples],
                })
            prompt += content.strip()
        elif role == "user":
            prompt += f"\n\n{content.strip()}\n\n"
    return samples


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", default="/root/autodl-tmp/memrag_new/multihop_oracle_sft_full_n1000_top3_doc220/train.jsonl")
    parser.add_argument("--valid-source", default="/root/autodl-tmp/memrag_new/multihop_oracle_sft_full_n1000_top3_doc220/valid.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/memgen_state_experience_sft_n1000")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--valid-limit", type=int, default=0)
    parser.add_argument("--experience-topn", type=int, default=4)
    parser.add_argument("--max-state-chars", type=int, default=2200)
    args = parser.parse_args()

    train_rows = list(iter_jsonl(args.train_source, args.train_limit))
    valid_rows = list(iter_jsonl(args.valid_source, args.valid_limit))
    pool = [pool_item(row) for row in train_rows]
    pool = [x for x in pool if len(x.get("searches", [])) >= 2]

    train_samples = []
    for row in train_rows:
        train_samples.extend(make_state_samples(row, pool, args))
    valid_samples = []
    for row in valid_rows:
        valid_samples.extend(make_state_samples(row, pool, args))

    out = Path(args.output_dir)
    write_jsonl(out / "train.jsonl", train_samples)
    write_jsonl(out / "valid.jsonl", valid_samples)
    stats = {
        "train_source_rows": len(train_rows),
        "valid_source_rows": len(valid_rows),
        "experience_pool_rows": len(pool),
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "train_action_counts": {k: sum(s["action_type"] == k for s in train_samples) for k in ["search", "answer"]},
        "valid_action_counts": {k: sum(s["action_type"] == k for s in valid_samples) for k in ["search", "answer"]},
        "args": vars(args),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STATS", json.dumps(stats, ensure_ascii=False))
    if train_samples:
        print("SAMPLE", json.dumps(train_samples[0], ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
