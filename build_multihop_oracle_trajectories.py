#!/usr/bin/env python3
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests

try:
    from flashrag.pipeline.reasoning_pipeline import SearchR1Pipeline
    USER_PROMPT = SearchR1Pipeline.user_prompt
except Exception:
    USER_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}"""


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def iter_jsonl(path: Path, limit: int = 0, skip: int = 0) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        seen = 0
        emitted = 0
        for line in f:
            if not line.strip():
                continue
            if seen < skip:
                seen += 1
                continue
            item = json.loads(line)
            emitted += 1
            yield item
            if limit and emitted >= limit:
                break


def supporting_titles(item: Dict) -> List[str]:
    sf = item.get("metadata", {}).get("supporting_facts", {})
    titles = sf.get("title", []) if isinstance(sf, dict) else []
    out, seen = [], set()
    for title in titles:
        key = normalize_text(title)
        if key and key not in seen:
            out.append(str(title))
            seen.add(key)
    return out


def golden_answer(item: Dict) -> str:
    answers = item.get("golden_answers") or item.get("answer") or []
    if isinstance(answers, list) and answers:
        return str(answers[0])
    return str(answers)


def title_from_doc(doc: Dict) -> str:
    contents = doc.get("contents", "") or doc.get("document", {}).get("contents", "")
    return contents.split("\n", 1)[0].strip().strip('"')


def contents_from_doc(doc: Dict) -> str:
    return doc.get("contents", "") or doc.get("document", {}).get("contents", "")


def retrieve(session: requests.Session, url: str, query: str, topk: int, cache: Dict) -> List[Dict]:
    key = f"{topk}\t{query}"
    if key in cache:
        return cache[key]
    last_err = None
    for attempt in range(3):
        try:
            resp = session.post(url, json={"queries": [query], "topk": topk, "return_scores": True}, timeout=180)
            resp.raise_for_status()
            raw_hits = resp.json()["result"][0]
            docs = []
            for hit in raw_hits:
                d = hit.get("document", hit)
                docs.append({"id": d.get("id"), "contents": d.get("contents", ""), "score": hit.get("score")})
            cache[key] = docs
            return docs
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"retrieval failed for {query!r}: {last_err}")


def docs_to_information(docs: List[Dict], max_doc_chars: int) -> str:
    text = ""
    for idx, doc in enumerate(docs, 1):
        contents = contents_from_doc(doc)
        title = title_from_doc(doc)
        body = contents.split("\n", 1)[1] if "\n" in contents else contents
        body = re.sub(r"\s+", " ", body).strip()[:max_doc_chars]
        text += f"Doc {idx}(Title: {title}) {body}\n"
    return f"<information>\n{text}</information>"


def retrieval_hits(docs: List[Dict], support_titles: List[str], answers: List[str]) -> Tuple[List[str], bool]:
    support_norm = {normalize_text(t): t for t in support_titles}
    answer_norms = [normalize_text(a) for a in answers if str(a).strip()]
    hit_titles, hit_answer = [], False
    for doc in docs:
        nt = normalize_text(title_from_doc(doc))
        nc = normalize_text(contents_from_doc(doc))
        for key, raw in support_norm.items():
            if key and (nt == key or key in nt or nt in key) and raw not in hit_titles:
                hit_titles.append(raw)
        if any(a and a in nc for a in answer_norms):
            hit_answer = True
    return hit_titles, hit_answer


def build_messages(item: Dict, dataset: str, session: requests.Session, args, cache: Dict) -> Tuple[Dict, Dict]:
    question = item["question"]
    support = supporting_titles(item)
    answer = golden_answer(item)
    if not support or not answer:
        raise ValueError("missing supporting titles or answer")

    messages = [{"role": "user", "content": USER_PROMPT.format(question=question)}]
    retrieved = set()
    events = []

    for turn, title in enumerate(support[: args.max_hops], 1):
        if turn == 1:
            thought = f"I need evidence about {title} before answering the question."
        else:
            thought = f"I need the next supporting evidence about {title} to complete the multi-hop reasoning."
        messages.append({"role": "assistant", "content": f"<think> {thought} </think>\n<search> {title} </search>"})
        docs = retrieve(session, args.search_url, title, args.topk, cache)
        hit_titles, hit_answer = retrieval_hits(docs, support, item.get("golden_answers", [answer]))
        for h in hit_titles:
            retrieved.add(normalize_text(h))
        events.append({
            "turn": turn,
            "query": title,
            "hit_support_titles": hit_titles,
            "hit_answer": hit_answer,
            "top_titles": [title_from_doc(d) for d in docs],
        })
        messages.append({"role": "user", "content": docs_to_information(docs, args.max_doc_chars)})

    messages.append({"role": "assistant", "content": f"<think> I have gathered the supporting evidence and can answer concisely. </think>\n<answer> {answer} </answer>"})

    row = {
        "id": item.get("id"),
        "dataset": dataset,
        "question": question,
        "golden_answers": item.get("golden_answers", [answer]),
        "supporting_titles": support,
        "messages": messages,
    }
    stats = {
        "id": item.get("id"),
        "dataset": dataset,
        "support_count": len(support),
        "support_coverage": len(retrieved) / len({normalize_text(t) for t in support}),
        "events": events,
    }
    return row, stats


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_split(name: str, specs: List[Tuple[str, Path, int, int]], session: requests.Session, args, cache: Dict):
    rows, stats, errors = [], [], []
    rejected = []
    scan_counts = {}
    for dataset, path, target, skip in specs:
        accepted = 0
        scanned = 0
        for item in iter_jsonl(path, limit=0, skip=skip):
            if args.max_scan_per_dataset and scanned >= args.max_scan_per_dataset:
                break
            scanned += 1
            try:
                row, stat = build_messages(item, dataset, session, args, cache)
                if args.require_full_support and stat["support_coverage"] < 0.999:
                    rejected.append({
                        "dataset": dataset,
                        "id": item.get("id"),
                        "support_coverage": stat["support_coverage"],
                        "events": stat["events"],
                    })
                    continue
                rows.append(row)
                stats.append(stat)
                accepted += 1
                if len(rows) % 50 == 0:
                    cov = sum(s["support_coverage"] for s in stats) / len(stats)
                    print(
                        f"{name}: built {len(rows)} rows, dataset={dataset}, accepted={accepted}/{scanned}, "
                        f"avg_support_coverage={cov:.3f}, rejected={len(rejected)}, cache={len(cache)}",
                        flush=True,
                    )
                if accepted >= target:
                    break
            except Exception as exc:
                errors.append({"dataset": dataset, "id": item.get("id"), "error": str(exc)})
        scan_counts[dataset] = {"accepted": accepted, "scanned": scanned, "target": target}
        if accepted < target:
            print(f"WARNING {name}: dataset={dataset} accepted only {accepted}/{target} after scanning {scanned}", flush=True)
    return rows, stats, errors, rejected, scan_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot-train", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/train.jsonl")
    parser.add_argument("--twiki-train", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/train.jsonl")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/multihop_oracle_sft_n1000")
    parser.add_argument("--train-per-dataset", type=int, default=500)
    parser.add_argument("--valid-per-dataset", type=int, default=50)
    parser.add_argument("--valid-skip", type=int, default=500)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-doc-chars", type=int, default=450)
    parser.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--require-full-support", action="store_true")
    parser.add_argument("--max-scan-per-dataset", type=int, default=5000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    session = requests.Session()
    cache = {}

    train_specs = [
        ("hotpotqa", Path(args.hotpot_train), args.train_per_dataset, 0),
        ("2wikimultihopqa", Path(args.twiki_train), args.train_per_dataset, 0),
    ]
    valid_specs = [
        ("hotpotqa", Path(args.hotpot_train), args.valid_per_dataset, args.valid_skip),
        ("2wikimultihopqa", Path(args.twiki_train), args.valid_per_dataset, args.valid_skip),
    ]
    train_rows, train_stats, train_errors, train_rejected, train_scan_counts = build_split("train", train_specs, session, args, cache)
    valid_rows, valid_stats, valid_errors, valid_rejected, valid_scan_counts = build_split("valid", valid_specs, session, args, cache)

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "valid.jsonl", valid_rows)
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "train_n": len(train_rows),
        "valid_n": len(valid_rows),
        "train_avg_support_coverage": sum(s["support_coverage"] for s in train_stats) / len(train_stats) if train_stats else None,
        "valid_avg_support_coverage": sum(s["support_coverage"] for s in valid_stats) / len(valid_stats) if valid_stats else None,
        "cache_size": len(cache),
        "train_rejected_n": len(train_rejected),
        "valid_rejected_n": len(valid_rejected),
        "train_scan_counts": train_scan_counts,
        "valid_scan_counts": valid_scan_counts,
        "errors": train_errors + valid_errors,
        "rejected_sample": (train_rejected + valid_rejected)[:20],
        "train_stats_sample": train_stats[:10],
        "valid_stats_sample": valid_stats[:10],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False))
    print("OUTPUT_DIR", out)


if __name__ == "__main__":
    main()
