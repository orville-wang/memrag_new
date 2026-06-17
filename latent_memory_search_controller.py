#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import pickle
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import requests
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.decomposition import TruncatedSVD
from transformers import AutoModelForCausalLM, AutoTokenizer

SEARCH_R1_URL = os.environ.get("SEARCH_R1_URL", "http://127.0.0.1:8000/retrieve")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "from", "by", "with", "was", "were",
    "is", "are", "who", "what", "which", "when", "where", "did", "do", "does", "same", "film", "movie",
    "director", "directed", "mother", "father", "country", "nationality", "born", "first", "second", "between",
}
CONNECTORS = {"of", "the", "and", "in", "for", "de", "la", "le", "du", "da", "di", "von", "van", "&", "'s"}
ANSWER_ACTION = "__ANSWER__"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_key(text: str) -> str:
    return normalize_text(text)


def clean_query(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" `\"'.,;:")
    text = re.sub(r"\((Film|film)\)", "(film)", text)
    return text[:180]


def token_set(text: str) -> set:
    return {t for t in normalize_text(text).split() if t and t not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def iter_jsonl(path: str, limit: int = 0, skip: int = 0) -> Iterable[Dict]:
    emitted = 0
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if seen < skip:
                seen += 1
                continue
            yield json.loads(line)
            emitted += 1
            if limit and emitted >= limit:
                break


def load_split(path: str, limit: int, dataset: str) -> List[Dict]:
    rows = []
    for item in iter_jsonl(path, limit=limit):
        item = dict(item)
        item["dataset"] = dataset
        rows.append(item)
    return rows


def supporting_titles(item: Dict) -> List[str]:
    sf = item.get("metadata", {}).get("supporting_facts", {})
    titles = sf.get("title", []) if isinstance(sf, dict) else []
    out, seen = [], set()
    for title in titles:
        title = str(title)
        key = norm_key(title)
        if key and key not in seen:
            out.append(title)
            seen.add(key)
    return out


def golden_answers(item: Dict) -> List[str]:
    answers = item.get("golden_answers") or item.get("answer") or []
    if isinstance(answers, list):
        return [str(x) for x in answers]
    return [str(answers)]


def context_titles(item: Dict) -> List[str]:
    ctx = item.get("metadata", {}).get("context", {})
    titles = ctx.get("title", []) if isinstance(ctx, dict) else []
    out, seen = [], set()
    for t in titles:
        key = norm_key(t)
        if key and key not in seen:
            out.append(str(t))
            seen.add(key)
    return out


def title_from_doc(doc: Dict) -> str:
    contents = doc.get("contents", "") or doc.get("document", {}).get("contents", "")
    return contents.split("\n", 1)[0].strip().strip('"')


def body_from_doc(doc: Dict) -> str:
    contents = doc.get("contents", "") or doc.get("document", {}).get("contents", "")
    if "\n" in contents:
        contents = contents.split("\n", 1)[1]
    return re.sub(r"\s+", " ", contents).strip()


def retrieve(session: requests.Session, query: str, topk: int, cache: Dict[str, List[Dict]]) -> List[Dict]:
    query = clean_query(query)
    key = f"{topk}\t{query}"
    if key in cache:
        return cache[key]
    last_err = None
    for attempt in range(3):
        try:
            resp = session.post(SEARCH_R1_URL, json={"queries": [query], "topk": topk, "return_scores": True}, timeout=180)
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
    raise RuntimeError(f"retrieve failed for {query!r}: {last_err}")


def doc_hits(docs: List[Dict], support: List[str], answers: List[str]) -> Dict:
    support_map = {norm_key(t): t for t in support}
    answer_norms = [norm_key(a) for a in answers if str(a).strip()]
    hit_support, top_titles = [], []
    hit_answer = False
    for doc in docs:
        title = title_from_doc(doc)
        top_titles.append(title)
        nt = norm_key(title)
        nc = norm_key(doc.get("contents", ""))
        for key, raw in support_map.items():
            if key and (nt == key or key in nt or nt in key) and raw not in hit_support:
                hit_support.append(raw)
        if any(a and a in nc for a in answer_norms):
            hit_answer = True
    return {"hit_support_titles": hit_support, "hit_answer": hit_answer, "top_titles": top_titles}


def final_answer_hit(answer: str, answers: List[str]) -> bool:
    na = norm_key(answer)
    for g in answers:
        ng = norm_key(g)
        if ng and (ng in na or na in ng):
            return True
    return False


def doc_view(docs: List[Dict], max_snippet_chars: int = 360) -> List[Dict]:
    out = []
    for d in docs:
        out.append({"title": title_from_doc(d), "snippet": body_from_doc(d)[:max_snippet_chars], "score": d.get("score")})
    return out


def extract_entities(text: str, max_items: int = 24) -> List[str]:
    text = re.sub(r"\s+", " ", str(text))
    found = []

    for pat in [r'"([^"]{3,90})"', r"'([^']{3,90})'", r"\b([A-Z][A-Za-z0-9À-ÖØ-öø-ÿ'’.-]+(?:\s+\([^)]+\))?)"]:
        for m in re.finditer(pat, text):
            val = clean_query(m.group(1))
            if val:
                found.append(val)

    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9][A-Za-zÀ-ÖØ-öø-ÿ0-9'’.-]*|\([^)]+\)", text)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        starts = tok[:1].isupper() or tok[:1].isdigit() or tok.startswith("(")
        if not starts:
            i += 1
            continue
        span = [tok]
        j = i + 1
        while j < len(tokens):
            nt = tokens[j]
            low = nt.lower()
            if nt[:1].isupper() or nt[:1].isdigit() or nt.startswith("(") or low in CONNECTORS:
                span.append(nt)
                j += 1
            else:
                break
        if len(span) >= 1:
            cand = clean_query(" ".join(span))
            if cand and len(norm_key(cand)) >= 3:
                found.append(cand)
        i = max(j, i + 1)

    # Add compact parenthetical-normalized variants often used in 2Wiki titles.
    variants = []
    for cand in found:
        variants.append(cand)
        variants.append(re.sub(r"\(([^)]*)\)", lambda m: "(" + m.group(1).lower() + ")", cand))
        variants.append(re.sub(r"\s+\((film|movie)\)", " (film)", cand, flags=re.I))

    out, seen = [], set()
    bad = {"Which", "What", "Who", "Were", "Was", "Are", "Is", "Did", "Do", "Does", "When", "Where"}
    for cand in variants:
        cand = clean_query(cand)
        if not cand or cand in bad:
            continue
        key = norm_key(cand)
        if not key or key in seen:
            continue
        if len(key.split()) == 1 and key in STOPWORDS:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= max_items:
            break
    return out


def relation_terms(question: str, max_terms: int = 4) -> List[str]:
    toks = [t for t in normalize_text(question).split() if t not in STOPWORDS and len(t) > 2]
    return toks[:max_terms]


def state_text(question: str, previous_queries: List[str], docs: List[Dict], step: int) -> str:
    lines = [f"Question: {question}", f"Step: {step}"]
    if previous_queries:
        lines.append("Previous queries: " + " | ".join(previous_queries[-4:]))
    if docs:
        lines.append("Retrieved titles: " + " | ".join([d.get("title", "") for d in docs[-8:]]))
        snippets = []
        for d in docs[-5:]:
            if d.get("snippet"):
                snippets.append(f"{d.get('title')}: {d.get('snippet')[:260]}")
        if snippets:
            lines.append("Retrieved snippets: " + " || ".join(snippets))
    return "\n".join(lines)


def runtime_candidates(question: str, previous_queries: List[str], docs: List[Dict], max_candidates: int = 18) -> List[str]:
    cands = []
    if not previous_queries:
        cands.append(question)
    cands.extend(extract_entities(question, max_items=16))
    rel = relation_terms(question)
    doc_entities = []
    if docs:
        for d in docs[-8:]:
            if d.get("title"):
                doc_entities.append(d["title"])
            doc_entities.extend(extract_entities(d.get("snippet", ""), max_items=12))
    for ent in doc_entities:
        cands.append(ent)
        if rel:
            cands.append(clean_query(ent + " " + " ".join(rel[:2])))
    # Keep duplicates in the candidate universe only once; repeated query behavior is measured after selection.
    out, seen = [], set()
    for c in cands:
        c = clean_query(c)
        key = norm_key(c)
        if not key or key in seen:
            continue
        if len(key.split()) > 14:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_candidates:
            break
    return out


def outcome_for_search(
    session: requests.Session,
    cache: Dict[str, List[Dict]],
    query: str,
    support: List[str],
    answers: List[str],
    retrieved_support_norm: set,
    previous_queries: List[str],
    topk: int,
) -> Dict:
    docs = retrieve(session, query, topk, cache)
    hits = doc_hits(docs, support, answers)
    new_hits = [t for t in hits["hit_support_titles"] if norm_key(t) not in retrieved_support_norm]
    duplicate = norm_key(query) in {norm_key(q) for q in previous_queries}
    no_hit = (not hits["hit_support_titles"]) and (not hits["hit_answer"])
    missing_count = max(1, len({norm_key(t) for t in support} - set(retrieved_support_norm)))
    score = (len(new_hits) / missing_count) + (0.15 if hits["hit_answer"] else 0.0) - (0.35 if duplicate else 0.0) - (0.25 if no_hit else 0.0)
    score -= max(0, len(query.split()) - 8) * 0.015
    return {
        "docs": docs,
        "hit_support_titles": hits["hit_support_titles"],
        "new_hit_support_titles": new_hits,
        "hit_answer": hits["hit_answer"],
        "top_titles": hits["top_titles"],
        "duplicate": duplicate,
        "no_hit": no_hit,
        "outcome_score": score,
        "label": bool(new_hits) and not duplicate,
    }


def make_action_row(
    state_id: str,
    dataset: str,
    split: str,
    item_id: str,
    question: str,
    step: int,
    previous_queries: List[str],
    state_docs: List[Dict],
    support: List[str],
    answers: List[str],
    retrieved_support_norm: set,
    action_type: str,
    action_text: str,
    session: requests.Session,
    cache: Dict[str, List[Dict]],
    topk: int,
) -> Dict:
    complete = len({norm_key(t) for t in support} - set(retrieved_support_norm)) == 0
    st = state_text(question, previous_queries, state_docs, step)
    row = {
        "state_id": state_id,
        "dataset": dataset,
        "split": split,
        "item_id": item_id,
        "question": question,
        "step": step,
        "previous_queries": list(previous_queries),
        "retrieved_titles": [d.get("title", "") for d in state_docs[-8:]],
        "state_text": st,
        "supporting_titles": support,
        "golden_answers": answers,
        "retrieved_support_titles": [t for t in support if norm_key(t) in retrieved_support_norm],
        "action_type": action_type,
        "action_text": action_text,
    }
    if action_type == "answer":
        row.update({
            "label": complete,
            "outcome_score": 1.0 if complete else -0.45,
            "duplicate": False,
            "no_hit": False,
            "hit_answer": False,
            "hit_support_titles": [],
            "new_hit_support_titles": [],
            "top_titles": [],
        })
    else:
        out = outcome_for_search(session, cache, action_text, support, answers, retrieved_support_norm, previous_queries, topk)
        row.update({k: v for k, v in out.items() if k != "docs"})
        if complete:
            row["label"] = False
            row["outcome_score"] = min(row["outcome_score"], -0.10)
    return row


def build_state_rows_for_item(
    item: Dict,
    dataset: str,
    split: str,
    session: requests.Session,
    cache: Dict[str, List[Dict]],
    global_titles: List[str],
    topk: int,
    max_hops: int,
    negatives_per_state: int,
    rng: random.Random,
) -> List[Dict]:
    question = item["question"]
    support = supporting_titles(item)
    answers = golden_answers(item)
    if len(support) < 2:
        return []
    c_titles = [t for t in context_titles(item) if norm_key(t) not in {norm_key(s) for s in support}]
    state_docs: List[Dict] = []
    previous_queries: List[str] = []
    retrieved_support_norm: set = set()
    rows: List[Dict] = []

    for step in range(min(max_hops, len(support))):
        missing = [t for t in support if norm_key(t) not in retrieved_support_norm]
        if not missing:
            break
        positive_query = missing[0]
        state_id = f"{split}:{dataset}:{item.get('id')}:{step}"
        pool = []
        pool.append(question)
        pool.extend(extract_entities(question, max_items=18))
        pool.extend(previous_queries[-2:])
        pool.extend([d.get("title", "") for d in state_docs[-6:]])
        for d in state_docs[-5:]:
            pool.extend(extract_entities(d.get("snippet", ""), max_items=10))
        pool.extend(c_titles[:10])
        if global_titles:
            pool.extend(rng.sample(global_titles, k=min(4, len(global_titles))))

        action_rows = []
        # Force one oracle positive query candidate, then add hard negatives.
        action_rows.append(make_action_row(state_id, dataset, split, item.get("id"), question, step, previous_queries, state_docs, support, answers, retrieved_support_norm, "search", positive_query, session, cache, topk))
        seen = {norm_key(positive_query)}
        for cand in pool:
            cand = clean_query(cand)
            key = norm_key(cand)
            if not key or key in seen:
                continue
            seen.add(key)
            row = make_action_row(state_id, dataset, split, item.get("id"), question, step, previous_queries, state_docs, support, answers, retrieved_support_norm, "search", cand, session, cache, topk)
            action_rows.append(row)
        action_rows.append(make_action_row(state_id, dataset, split, item.get("id"), question, step, previous_queries, state_docs, support, answers, retrieved_support_norm, "answer", ANSWER_ACTION, session, cache, topk))

        positives = [r for r in action_rows if r["label"]]
        negatives = [r for r in action_rows if not r["label"]]
        if positives:
            positives = sorted(positives, key=lambda r: r["outcome_score"], reverse=True)[:1]
            # Harder negatives first: duplicate/no-hit/answer-too-early/high lexical overlap.
            negatives = sorted(negatives, key=lambda r: (r["action_type"] == "answer", r.get("duplicate", False), r.get("no_hit", False), jaccard(r["question"], r["action_text"])), reverse=True)
            rows.extend(positives + negatives[:negatives_per_state])

        # Advance the oracle state with the selected positive query.
        docs = retrieve(session, positive_query, topk, cache)
        hits = doc_hits(docs, support, answers)
        for t in hits["hit_support_titles"]:
            retrieved_support_norm.add(norm_key(t))
        previous_queries.append(positive_query)
        state_docs.extend(doc_view(docs))

    if len({norm_key(t) for t in support} - retrieved_support_norm) == 0:
        state_id = f"{split}:{dataset}:{item.get('id')}:complete"
        pool = []
        pool.extend(previous_queries[-3:])
        pool.extend([d.get("title", "") for d in state_docs[-8:]])
        pool.extend(rng.sample(global_titles, k=min(4, len(global_titles))))
        action_rows = [make_action_row(state_id, dataset, split, item.get("id"), question, len(previous_queries), previous_queries, state_docs, support, answers, retrieved_support_norm, "answer", ANSWER_ACTION, session, cache, topk)]
        seen = set()
        for cand in pool:
            key = norm_key(cand)
            if not key or key in seen:
                continue
            seen.add(key)
            action_rows.append(make_action_row(state_id, dataset, split, item.get("id"), question, len(previous_queries), previous_queries, state_docs, support, answers, retrieved_support_norm, "search", cand, session, cache, topk))
        positives = [r for r in action_rows if r["label"]]
        negatives = [r for r in action_rows if not r["label"]]
        rows.extend(positives[:1] + negatives[:negatives_per_state])
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def build_data(args) -> Dict:
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "retrieval_cache.pkl"
    if cache_path.exists() and not args.rebuild_cache:
        cache = pickle.loads(cache_path.read_bytes())
    else:
        cache = {}
    session = requests.Session()

    train_specs = [("hotpotqa", args.hotpot_train, args.train_per_dataset), ("2wikimultihopqa", args.twiki_train, args.train_per_dataset)]
    dev_specs = [("hotpotqa", args.hotpot_dev, args.dev_per_dataset), ("2wikimultihopqa", args.twiki_dev, args.dev_per_dataset)]

    all_train_items = []
    for dataset, path, target in train_specs:
        all_train_items.extend(load_split(path, target * 3, dataset))
    global_titles = []
    seen = set()
    for item in all_train_items:
        for t in supporting_titles(item) + context_titles(item):
            key = norm_key(t)
            if key and key not in seen:
                seen.add(key)
                global_titles.append(t)

    split_rows = {}
    split_stats = {}
    for split, specs in [("train", train_specs), ("dev", dev_specs)]:
        rows = []
        stats = {}
        for dataset, path, target in specs:
            accepted = 0
            scanned = 0
            for item in iter_jsonl(path):
                scanned += 1
                item_rows = build_state_rows_for_item(item, dataset, split, session, cache, global_titles, args.topk, args.max_hops, args.negatives_per_state, rng)
                if item_rows:
                    rows.extend(item_rows)
                    accepted += 1
                    if accepted % 25 == 0:
                        pos = sum(r["label"] for r in rows)
                        print(f"BUILD {split} {dataset}: accepted_items={accepted}/{scanned} rows={len(rows)} positives={pos} cache={len(cache)}", flush=True)
                if accepted >= target:
                    break
            stats[dataset] = {"accepted_items": accepted, "scanned_items": scanned, "target_items": target}
        split_rows[split] = rows
        split_stats[split] = stats
        write_jsonl(out_dir / f"{split}_state_actions.jsonl", rows)

    cache_path.write_bytes(pickle.dumps(cache))
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": vars(args),
        "stats": split_stats,
        "train_rows": len(split_rows["train"]),
        "train_states": len({r["state_id"] for r in split_rows["train"]}),
        "train_positive_rows": sum(r["label"] for r in split_rows["train"]),
        "dev_rows": len(split_rows["dev"]),
        "dev_states": len({r["state_id"] for r in split_rows["dev"]}),
        "dev_positive_rows": sum(r["label"] for r in split_rows["dev"]),
        "retrieval_cache_size": len(cache),
    }
    (out_dir / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("DATA_SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


@dataclass
class ControllerBundle:
    use_memory: bool
    text_vectorizer: TfidfVectorizer
    clf: LogisticRegression
    memory_state_vectorizer: Optional[TfidfVectorizer] = None
    memory_svd: Optional[TruncatedSVD] = None
    memory_embeddings: Optional[np.ndarray] = None
    memory_records: Optional[List[Dict]] = None

    def retrieve_memory(self, st: str, k: int = 5) -> List[Dict]:
        if not self.use_memory or not self.memory_records or self.memory_embeddings is None:
            return []
        X = self.memory_state_vectorizer.transform([st])
        z = self.memory_svd.transform(X)
        z = sk_normalize(z)
        sims = np.asarray(z @ self.memory_embeddings.T).ravel()
        if sims.size == 0:
            return []
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            rec = dict(self.memory_records[int(i)])
            rec["memory_similarity"] = float(sims[int(i)])
            out.append(rec)
        return out

    def memory_context(self, st: str, k: int = 5) -> str:
        memories = self.retrieve_memory(st, k=k)
        if not memories:
            return ""
        lines = []
        for i, m in enumerate(memories, 1):
            lines.append(
                f"Memory {i}: step={m.get('step')} action={m.get('action_type')} query={m.get('action_text')} "
                f"outcome={m.get('outcome_score'):.3f} sim={m.get('memory_similarity'):.3f}"
            )
        return "\n".join(lines)

    def row_text(self, row: Dict) -> str:
        base = row["state_text"] + f"\nCandidate action: {row['action_type']} {row['action_text']}"
        if self.use_memory:
            mem = self.memory_context(row["state_text"])
            if mem:
                return row["state_text"] + "\nRetrieved experience memory:\n" + mem + f"\nCandidate action: {row['action_type']} {row['action_text']}"
        return base

    def numeric_features(self, rows: List[Dict]) -> np.ndarray:
        feats = []
        for row in rows:
            q = row.get("question", "")
            action = row.get("action_text", "")
            prev = row.get("previous_queries", []) or []
            retrieved_titles = row.get("retrieved_titles", []) or []
            is_answer = row.get("action_type") == "answer"
            duplicate = (not is_answer) and norm_key(action) in {norm_key(x) for x in prev}
            in_question = (not is_answer) and norm_key(action) and norm_key(action) in norm_key(q)
            is_retrieved_title = (not is_answer) and norm_key(action) in {norm_key(t) for t in retrieved_titles}
            mem_max_sim = mem_mean_sim = mem_query_max_j = mem_query_mean_j = mem_outcome_mean = mem_action_match = 0.0
            if self.use_memory:
                memories = self.retrieve_memory(row["state_text"])
                if memories:
                    sims = [m.get("memory_similarity", 0.0) for m in memories]
                    mem_max_sim = max(sims)
                    mem_mean_sim = sum(sims) / len(sims)
                    js = [jaccard(action, m.get("action_text", "")) for m in memories]
                    mem_query_max_j = max(js)
                    mem_query_mean_j = sum(js) / len(js)
                    mem_outcome_mean = sum(m.get("outcome_score", 0.0) for m in memories) / len(memories)
                    mem_action_match = sum(1 for m in memories if m.get("action_type") == row.get("action_type")) / len(memories)
            feats.append([
                1.0 if is_answer else 0.0,
                1.0 if row.get("action_type") == "search" else 0.0,
                float(row.get("step", 0)),
                float(len(prev)),
                float(len(retrieved_titles)),
                float(len(action.split())) if not is_answer else 0.0,
                1.0 if duplicate else 0.0,
                1.0 if in_question else 0.0,
                1.0 if is_retrieved_title else 0.0,
                jaccard(q, action) if not is_answer else 0.0,
                mem_max_sim,
                mem_mean_sim,
                mem_query_max_j,
                mem_query_mean_j,
                mem_outcome_mean,
                mem_action_match,
            ])
        return np.asarray(feats, dtype=np.float32)

    def transform_rows(self, rows: List[Dict]):
        texts = [self.row_text(r) for r in rows]
        X_text = self.text_vectorizer.transform(texts)
        X_num = sparse.csr_matrix(self.numeric_features(rows))
        return sparse.hstack([X_text, X_num], format="csr")

    def predict_proba(self, rows: List[Dict]) -> np.ndarray:
        X = self.transform_rows(rows)
        return self.clf.predict_proba(X)[:, 1]


def make_memory(records: List[Dict], n_components: int = 64):
    memory_records = []
    for r in records:
        if r.get("label") and r.get("action_type") == "search":
            memory_records.append({
                "state_id": r["state_id"],
                "question": r["question"],
                "state_text": r["state_text"],
                "step": r.get("step", 0),
                "action_type": r["action_type"],
                "action_text": r["action_text"],
                "outcome_score": float(r.get("outcome_score", 0.0)),
            })
    texts = [m["state_text"] for m in memory_records]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)
    comps = max(2, min(n_components, X.shape[0] - 1, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=comps, random_state=13)
    Z = sk_normalize(svd.fit_transform(X))
    return vec, svd, Z, memory_records


def fit_controller(train_rows: List[Dict], use_memory: bool) -> ControllerBundle:
    bundle = ControllerBundle(use_memory=use_memory, text_vectorizer=None, clf=None)
    if use_memory:
        vec, svd, emb, mem = make_memory(train_rows)
        bundle.memory_state_vectorizer = vec
        bundle.memory_svd = svd
        bundle.memory_embeddings = emb
        bundle.memory_records = mem
    texts = [bundle.row_text(r) for r in train_rows]
    text_vec = TfidfVectorizer(max_features=120000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_text = text_vec.fit_transform(texts)
    bundle.text_vectorizer = text_vec
    X_num = sparse.csr_matrix(bundle.numeric_features(train_rows))
    X = sparse.hstack([X_text, X_num], format="csr")
    y = np.asarray([1 if r.get("label") else 0 for r in train_rows], dtype=np.int32)
    C = 0.5 if use_memory else 2.0
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=C, solver="liblinear", random_state=17)
    clf.fit(X, y)
    bundle.clf = clf
    return bundle


def group_by_state(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(r["state_id"], []).append(r)
    return groups


def offline_eval(bundle: ControllerBundle, rows: List[Dict]) -> Dict:
    groups = group_by_state(rows)
    top1_good = 0
    top1_answer_when_complete = 0
    answer_false_positive = 0
    complete_states = 0
    incomplete_states = 0
    all_scores, all_labels = [], []
    for state_id, cand_rows in groups.items():
        probs = bundle.predict_proba(cand_rows)
        labels = [bool(r["label"]) for r in cand_rows]
        all_scores.extend(probs.tolist())
        all_labels.extend([1 if x else 0 for x in labels])
        best_i = int(np.argmax(probs))
        best = cand_rows[best_i]
        if labels[best_i]:
            top1_good += 1
        complete = any(r["action_type"] == "answer" and r["label"] for r in cand_rows)
        if complete:
            complete_states += 1
            if best["action_type"] == "answer":
                top1_answer_when_complete += 1
        else:
            incomplete_states += 1
            if best["action_type"] == "answer":
                answer_false_positive += 1
    auc = None
    ap = None
    if len(set(all_labels)) > 1:
        auc = float(roc_auc_score(all_labels, all_scores))
        ap = float(average_precision_score(all_labels, all_scores))
    n = len(groups)
    return {
        "states": n,
        "top1_positive_rate": top1_good / n if n else 0,
        "complete_states": complete_states,
        "answer_top1_on_complete_rate": top1_answer_when_complete / complete_states if complete_states else 0,
        "incomplete_states": incomplete_states,
        "answer_false_positive_rate": answer_false_positive / incomplete_states if incomplete_states else 0,
        "roc_auc": auc,
        "average_precision": ap,
    }


def train_models(args) -> Dict:
    out_dir = Path(args.output_dir)
    train_rows = read_jsonl(out_dir / "train_state_actions.jsonl")
    dev_rows = read_jsonl(out_dir / "dev_state_actions.jsonl")
    models = {}
    summaries = {}
    for name, use_memory in [("no_memory_ranker", False), ("latent_memory_ranker", True)]:
        print(f"TRAIN {name} use_memory={use_memory}", flush=True)
        bundle = fit_controller(train_rows, use_memory=use_memory)
        train_summary = offline_eval(bundle, train_rows)
        dev_summary = offline_eval(bundle, dev_rows)
        models[name] = bundle
        summaries[name] = {"train": train_summary, "dev": dev_summary}
        joblib.dump(bundle, out_dir / f"{name}.joblib")
        print("OFFLINE", name, json.dumps(summaries[name], ensure_ascii=False), flush=True)
    (out_dir / "train_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    return summaries


def make_runtime_action_rows(
    item: Dict,
    state_id: str,
    step: int,
    previous_queries: List[str],
    state_docs: List[Dict],
    candidates: List[Tuple[str, str]],
) -> List[Dict]:
    question = item["question"]
    support = supporting_titles(item)
    answers = golden_answers(item)
    st = state_text(question, previous_queries, state_docs, step)
    return [{
        "state_id": state_id,
        "dataset": item.get("dataset"),
        "split": "online_eval",
        "item_id": item.get("id"),
        "question": question,
        "step": step,
        "previous_queries": list(previous_queries),
        "retrieved_titles": [d.get("title", "") for d in state_docs[-8:]],
        "state_text": st,
        "supporting_titles": support,
        "golden_answers": answers,
        "retrieved_support_titles": [],
        "action_type": "search" if typ == "search" else "answer",
        "action_text": text,
        "label": False,
        "outcome_score": 0.0,
    } for typ, text in candidates]


def answer_prompt(question: str, docs: List[Dict], max_docs: int = 8) -> str:
    evidence = []
    for i, d in enumerate(docs[-max_docs:], 1):
        evidence.append(f"Doc {i} Title: {d.get('title', '')}\n{d.get('snippet', '')[:500]}")
    ev = "\n\n".join(evidence) if evidence else "No retrieved evidence."
    return (
        "Answer the question concisely using the retrieved evidence. "
        "Return only the short answer, with no explanation.\n\n"
        f"Question: {question}\n\nRetrieved evidence:\n{ev}"
    )


def load_answer_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    return tok, model


def generate_answer(tokenizer, model, question: str, docs: List[Dict], max_new_tokens: int) -> str:
    prompt = answer_prompt(question, docs)
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=6144).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    ans = tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    ans = ans.strip().split("\n")[0].strip()
    ans = re.sub(r"^(answer\s*[:\-]\s*)", "", ans, flags=re.I).strip()
    return ans[:240]


def choose_heuristic(question: str, previous_queries: List[str], state_docs: List[Dict], turn: int) -> Tuple[str, str, float, List[Dict]]:
    if turn >= 3:
        return "answer", ANSWER_ACTION, 1.0, []
    cands = runtime_candidates(question, previous_queries, state_docs, max_candidates=18)
    if not cands:
        return "answer", ANSWER_ACTION, 1.0, []
    if not previous_queries:
        return "search", cands[0], 1.0, []
    prev_norm = {norm_key(q) for q in previous_queries}
    for c in cands:
        if norm_key(c) not in prev_norm:
            return "search", c, 1.0, []
    return "answer", ANSWER_ACTION, 1.0, []


def run_controller_on_item(
    item: Dict,
    method: str,
    bundle: Optional[ControllerBundle],
    session: requests.Session,
    cache: Dict[str, List[Dict]],
    topk: int,
    max_turns: int,
    answer_threshold: float,
    tokenizer,
    answer_model,
    answer_max_new_tokens: int,
) -> Dict:
    support = supporting_titles(item)
    answers = golden_answers(item)
    previous_queries: List[str] = []
    state_docs: List[Dict] = []
    retrieved_support = set()
    trajectory = []
    repeated_queries = 0
    no_hit_searches = 0
    first_support_hit_turn = None
    first_answer_hit_turn = None
    all_support_hit_turn = None
    finish_reason = "max_turn"

    for turn in range(1, max_turns + 1):
        if method == "heuristic":
            action_type, action_text, score, scored = choose_heuristic(item["question"], previous_queries, state_docs, turn)
        else:
            cand_queries = runtime_candidates(item["question"], previous_queries, state_docs, max_candidates=18)
            candidates = [("search", q) for q in cand_queries]
            candidates.append(("answer", ANSWER_ACTION))
            state_id = f"online:{method}:{item.get('dataset')}:{item.get('id')}:{turn}"
            rows = make_runtime_action_rows(item, state_id, turn - 1, previous_queries, state_docs, candidates)
            probs = bundle.predict_proba(rows)
            order = np.argsort(-probs)
            chosen_idx = int(order[0])
            chosen = rows[chosen_idx]
            action_type, action_text, score = chosen["action_type"], chosen["action_text"], float(probs[chosen_idx])
            if action_type == "answer" and score < answer_threshold:
                for idx in order:
                    if rows[int(idx)]["action_type"] == "search":
                        chosen_idx = int(idx)
                        chosen = rows[chosen_idx]
                        action_type, action_text, score = chosen["action_type"], chosen["action_text"], float(probs[chosen_idx])
                        break
            scored = [
                {"action_type": rows[int(i)]["action_type"], "action_text": rows[int(i)]["action_text"], "score": float(probs[int(i)])}
                for i in order[:6]
            ]

        event = {"turn": turn, "action_type": action_type, "action_text": action_text, "controller_score": score, "top_scored_actions": scored}
        if action_type == "answer":
            finish_reason = "answer"
            trajectory.append(event)
            break

        qn = norm_key(action_text)
        event["duplicate_query"] = qn in {norm_key(q) for q in previous_queries}
        if event["duplicate_query"]:
            repeated_queries += 1
        previous_queries.append(action_text)
        docs = retrieve(session, action_text, topk, cache)
        hits = doc_hits(docs, support, answers)
        for t in hits["hit_support_titles"]:
            retrieved_support.add(norm_key(t))
        if hits["hit_support_titles"] and first_support_hit_turn is None:
            first_support_hit_turn = turn
        if hits["hit_answer"] and first_answer_hit_turn is None:
            first_answer_hit_turn = turn
        if support and len(retrieved_support) >= len({norm_key(t) for t in support}) and all_support_hit_turn is None:
            all_support_hit_turn = turn
        if not hits["hit_support_titles"] and not hits["hit_answer"]:
            no_hit_searches += 1
        event.update(hits)
        event["docs"] = [{"title": title_from_doc(d), "score": d.get("score")} for d in docs]
        trajectory.append(event)
        state_docs.extend(doc_view(docs))

    final_answer = generate_answer(tokenizer, answer_model, item["question"], state_docs, answer_max_new_tokens)
    final_hit = final_answer_hit(final_answer, answers)
    searches = sum(1 for e in trajectory if e["action_type"] == "search")
    return {
        "dataset": item.get("dataset"),
        "id": item.get("id"),
        "question": item["question"],
        "golden_answers": answers,
        "supporting_titles": support,
        "method": method,
        "finish_reason": finish_reason,
        "final_answer": final_answer,
        "final_answer_hit": final_hit,
        "searches": searches,
        "repeated_queries": repeated_queries,
        "no_hit_searches": no_hit_searches,
        "support_coverage": len(retrieved_support) / len({norm_key(t) for t in support}) if support else None,
        "any_support_hit": first_support_hit_turn is not None,
        "first_support_hit_turn": first_support_hit_turn,
        "first_answer_hit_turn": first_answer_hit_turn,
        "all_support_hit_turn": all_support_hit_turn,
        "trajectory": trajectory,
    }


def avg(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def summarize_online(rows: List[Dict]) -> Dict:
    n = len(rows)
    searches = sum(r["searches"] for r in rows)
    return {
        "n": n,
        "final_answer_hit_rate": sum(r["final_answer_hit"] for r in rows) / n if n else 0.0,
        "avg_searches": avg([r["searches"] for r in rows]),
        "avg_support_coverage": avg([r["support_coverage"] for r in rows]),
        "any_support_hit_rate": sum(r["any_support_hit"] for r in rows) / n if n else 0.0,
        "all_support_hit_rate": sum(r["all_support_hit_turn"] is not None for r in rows) / n if n else 0.0,
        "avg_first_support_hit_turn_found_only": avg([r["first_support_hit_turn"] for r in rows]),
        "avg_first_answer_hit_turn_found_only": avg([r["first_answer_hit_turn"] for r in rows]),
        "duplicate_query_rate_per_search": sum(r["repeated_queries"] for r in rows) / searches if searches else 0.0,
        "no_hit_search_rate_per_search": sum(r["no_hit_searches"] for r in rows) / searches if searches else 0.0,
        "other_finish_rate": sum(r["finish_reason"] not in {"answer", "max_turn"} for r in rows) / n if n else 0.0,
        "max_turn_failure_rate": sum(r["finish_reason"] == "max_turn" for r in rows) / n if n else 0.0,
    }


def eval_online(args) -> Dict:
    out_dir = Path(args.output_dir)
    bundles = {
        "no_memory_ranker": joblib.load(out_dir / "no_memory_ranker.joblib"),
        "latent_memory_ranker": joblib.load(out_dir / "latent_memory_ranker.joblib"),
    }
    data = []
    data.extend(load_split(args.hotpot_dev, args.eval_per_dataset, "hotpotqa"))
    data.extend(load_split(args.twiki_dev, args.eval_per_dataset, "2wikimultihopqa"))

    cache_path = out_dir / "online_retrieval_cache.pkl"
    if cache_path.exists() and not args.rebuild_cache:
        cache = pickle.loads(cache_path.read_bytes())
    else:
        cache = {}
    session = requests.Session()
    print("Loading answer model", args.model, flush=True)
    tokenizer, answer_model = load_answer_model(args.model)

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": ["hotpotqa/dev", "2wikimultihopqa/dev"],
        "retriever": "Search-R1 BM25 over wiki18 via HTTP retrieve API",
        "settings": vars(args),
        "methods": {},
    }
    for method in ["heuristic", "no_memory_ranker", "latent_memory_ranker"]:
        rows = []
        bundle = bundles.get(method)
        print("ONLINE_EVAL", method, flush=True)
        for i, item in enumerate(data):
            row = run_controller_on_item(
                item, method, bundle, session, cache, args.topk, args.max_turns, args.answer_threshold,
                tokenizer, answer_model, args.answer_max_new_tokens,
            )
            rows.append(row)
            print(
                method, i, item.get("dataset"), item.get("id"), "searches", row["searches"],
                "cov", row["support_coverage"], "final_hit", row["final_answer_hit"],
                "first", row["first_support_hit_turn"], "nohit", row["no_hit_searches"], "dup", row["repeated_queries"],
                flush=True,
            )
            if (i + 1) % 10 == 0:
                cache_path.write_bytes(pickle.dumps(cache))
        result["methods"][method] = {"summary": summarize_online(rows), "rows": rows}
        (out_dir / args.online_output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("ONLINE_SUMMARY", method, json.dumps(result["methods"][method]["summary"], ensure_ascii=False), flush=True)

    del answer_model
    gc.collect()
    torch.cuda.empty_cache()
    cache_path.write_bytes(pickle.dumps(cache))
    result["summary"] = {k: v["summary"] for k, v in result["methods"].items()}
    (out_dir / args.online_output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("ONLINE_RESULT_PATH", str(out_dir / args.online_output), flush=True)
    print("ONLINE_SUMMARY_JSON", json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "train", "eval", "all"], default="all")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki")
    parser.add_argument("--hotpot-train", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/train.jsonl")
    parser.add_argument("--twiki-train", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/train.jsonl")
    parser.add_argument("--hotpot-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/dev.jsonl")
    parser.add_argument("--twiki-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/dev.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-per-dataset", type=int, default=300)
    parser.add_argument("--dev-per-dataset", type=int, default=60)
    parser.add_argument("--eval-per-dataset", type=int, default=25)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--negatives-per-state", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--answer-threshold", type=float, default=0.55)
    parser.add_argument("--answer-max-new-tokens", type=int, default=48)
    parser.add_argument("--online-output", default="latent_memory_controller_online_eval.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"), flush=True)
    print("SEARCH_R1_URL", SEARCH_R1_URL, flush=True)
    print("MODE", args.mode, flush=True)
    if args.mode in {"build", "all"}:
        build_data(args)
    if args.mode in {"train", "all"}:
        train_models(args)
    if args.mode in {"eval", "all"}:
        eval_online(args)


if __name__ == "__main__":
    main()
