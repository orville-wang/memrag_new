#!/usr/bin/env python3
import argparse
import gc
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import requests
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.decomposition import TruncatedSVD

import latent_memory_search_controller as lm


@dataclass
class NumericMemoryBundle(lm.ControllerBundle):
    def row_text(self, row: Dict) -> str:
        return row["state_text"] + f"\nCandidate action: {row['action_type']} {row['action_text']}"


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
    svd = TruncatedSVD(n_components=comps, random_state=23)
    Z = sk_normalize(svd.fit_transform(X))
    return vec, svd, Z, memory_records


def fit_numeric_controller(train_rows: List[Dict], use_memory: bool, c_value: float) -> NumericMemoryBundle:
    bundle = NumericMemoryBundle(use_memory=use_memory, text_vectorizer=None, clf=None)
    if use_memory:
        vec, svd, emb, mem = make_memory(train_rows)
        bundle.memory_state_vectorizer = vec
        bundle.memory_svd = svd
        bundle.memory_embeddings = emb
        bundle.memory_records = mem
    texts = [bundle.row_text(r) for r in train_rows]
    vec = TfidfVectorizer(max_features=120000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_text = vec.fit_transform(texts)
    bundle.text_vectorizer = vec
    X_num = sparse.csr_matrix(bundle.numeric_features(train_rows))
    X = sparse.hstack([X_text, X_num], format="csr")
    y = np.asarray([1 if r.get("label") else 0 for r in train_rows], dtype=np.int32)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=c_value, solver="liblinear", random_state=31)
    clf.fit(X, y)
    bundle.clf = clf
    return bundle


def group_by_state(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups = {}
    for r in rows:
        groups.setdefault(r["state_id"], []).append(r)
    return groups


def offline_search_eval(bundle: NumericMemoryBundle, rows: List[Dict]) -> Dict:
    groups = group_by_state(rows)
    total = pos_states = top1_pos = top1_nohit = 0
    scores = []
    for cand_rows in groups.values():
        probs = bundle.predict_proba(cand_rows)
        best_i = int(np.argmax(probs))
        best = cand_rows[best_i]
        total += 1
        scores.append(float(probs[best_i]))
        has_pos = any(r.get("label") for r in cand_rows)
        if has_pos:
            pos_states += 1
            if best.get("label"):
                top1_pos += 1
        if best.get("no_hit"):
            top1_nohit += 1
    return {
        "states": total,
        "states_with_positive_search": pos_states,
        "top1_positive_rate_on_positive_states": top1_pos / pos_states if pos_states else 0.0,
        "top1_no_hit_rate": top1_nohit / total if total else 0.0,
        "avg_best_search_score": sum(scores) / len(scores) if scores else None,
    }


def threshold_sweep(bundle: NumericMemoryBundle, rows: List[Dict], min_searches: int) -> List[Dict]:
    groups = group_by_state(rows)
    candidates = []
    for th in [x / 100 for x in range(5, 96, 5)]:
        search_states = good = nohit = stopped_need = stopped_complete = 0
        for cand_rows in groups.values():
            probs = bundle.predict_proba(cand_rows)
            best_i = int(np.argmax(probs))
            best = cand_rows[best_i]
            step = int(best.get("step", 0))
            has_pos = any(r.get("label") for r in cand_rows)
            should_force = step < min_searches
            do_search = should_force or float(probs[best_i]) >= th
            if do_search:
                search_states += 1
                if best.get("label"):
                    good += 1
                if best.get("no_hit"):
                    nohit += 1
            else:
                if has_pos:
                    stopped_need += 1
                else:
                    stopped_complete += 1
        objective = good - 0.6 * nohit - 0.4 * stopped_need - 0.05 * search_states
        candidates.append({
            "threshold": th,
            "objective": objective,
            "search_states": search_states,
            "good_searches": good,
            "nohit_searches": nohit,
            "stopped_when_positive_exists": stopped_need,
            "stopped_when_no_positive": stopped_complete,
        })
    return sorted(candidates, key=lambda x: x["objective"], reverse=True)


def fit_and_save(args) -> Dict:
    out_dir = Path(args.output_dir)
    train_all = lm.read_jsonl(out_dir / "train_state_actions.jsonl")
    dev_all = lm.read_jsonl(out_dir / "dev_state_actions.jsonl")
    train_search = [r for r in train_all if r.get("action_type") == "search"]
    dev_search = [r for r in dev_all if r.get("action_type") == "search"]
    specs = [
        ("no_memory_search_ranker_numeric", False, args.no_memory_c),
        ("numeric_memory_search_ranker", True, args.memory_c),
    ]
    summary = {}
    for name, use_memory, c_value in specs:
        print("TRAIN_NUMERIC", name, "use_memory", use_memory, "C", c_value, "rows", len(train_search), flush=True)
        bundle = fit_numeric_controller(train_search, use_memory=use_memory, c_value=c_value)
        summary[name] = {
            "train": offline_search_eval(bundle, train_search),
            "dev": offline_search_eval(bundle, dev_search),
            "threshold_sweep_top5": threshold_sweep(bundle, dev_search, args.min_searches)[:5],
        }
        joblib.dump(bundle, out_dir / f"{name}.joblib")
        print("NUMERIC_OFFLINE", name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    (out_dir / "numeric_memory_search_train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def choose_search(bundle, item, turn, previous_queries, state_docs, threshold, min_searches):
    cand_queries = lm.runtime_candidates(item["question"], previous_queries, state_docs, max_candidates=24)
    if not cand_queries:
        return "answer", lm.ANSWER_ACTION, 0.0, []
    rows = lm.make_runtime_action_rows(item, f"numeric_online:{item.get('dataset')}:{item.get('id')}:{turn}", turn - 1, previous_queries, state_docs, [("search", q) for q in cand_queries])
    probs = bundle.predict_proba(rows)
    order = np.argsort(-probs)
    prev = {lm.norm_key(q) for q in previous_queries}
    ranked = []
    chosen = None
    for idx in order:
        idx = int(idx)
        q = rows[idx]["action_text"]
        score = float(probs[idx])
        ranked.append({"action_type": "search", "action_text": q, "score": score})
        if chosen is None and lm.norm_key(q) not in prev:
            chosen = (q, score)
    if chosen is None:
        idx = int(order[0])
        chosen = (rows[idx]["action_text"], float(probs[idx]))
    if turn > min_searches and chosen[1] < threshold:
        return "answer", lm.ANSWER_ACTION, chosen[1], ranked[:8]
    return "search", chosen[0], chosen[1], ranked[:8]


def run_item(item, method, bundle, session, cache, args, tokenizer, answer_model):
    support = lm.supporting_titles(item)
    answers = lm.golden_answers(item)
    previous_queries, state_docs, trajectory = [], [], []
    retrieved_support = set()
    repeated = nohit = 0
    first_support = first_answer = all_support = None
    finish_reason = "max_turn"
    for turn in range(1, args.max_turns + 1):
        if method == "heuristic_2search":
            action_type, action_text, score, ranked = lm.choose_heuristic(item["question"], previous_queries, state_docs, turn)
        else:
            action_type, action_text, score, ranked = choose_search(bundle, item, turn, previous_queries, state_docs, args.search_threshold, args.min_searches)
        event = {"turn": turn, "action_type": action_type, "action_text": action_text, "controller_score": score, "top_scored_actions": ranked}
        if action_type == "answer":
            finish_reason = "answer"
            trajectory.append(event)
            break
        qn = lm.norm_key(action_text)
        event["duplicate_query"] = qn in {lm.norm_key(q) for q in previous_queries}
        if event["duplicate_query"]:
            repeated += 1
        previous_queries.append(action_text)
        docs = lm.retrieve(session, action_text, args.topk, cache)
        hits = lm.doc_hits(docs, support, answers)
        for t in hits["hit_support_titles"]:
            retrieved_support.add(lm.norm_key(t))
        if hits["hit_support_titles"] and first_support is None:
            first_support = turn
        if hits["hit_answer"] and first_answer is None:
            first_answer = turn
        if support and len(retrieved_support) >= len({lm.norm_key(t) for t in support}) and all_support is None:
            all_support = turn
        if not hits["hit_support_titles"] and not hits["hit_answer"]:
            nohit += 1
        event.update(hits)
        event["docs"] = [{"title": lm.title_from_doc(d), "score": d.get("score")} for d in docs]
        trajectory.append(event)
        state_docs.extend(lm.doc_view(docs))
    final_answer = lm.generate_answer(tokenizer, answer_model, item["question"], state_docs, args.answer_max_new_tokens)
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
        "final_answer_hit": lm.final_answer_hit(final_answer, answers),
        "searches": searches,
        "repeated_queries": repeated,
        "no_hit_searches": nohit,
        "support_coverage": len(retrieved_support) / len({lm.norm_key(t) for t in support}) if support else None,
        "any_support_hit": first_support is not None,
        "first_support_hit_turn": first_support,
        "first_answer_hit_turn": first_answer,
        "all_support_hit_turn": all_support,
        "trajectory": trajectory,
    }


def eval_online(args):
    out_dir = Path(args.output_dir)
    bundles = {
        "no_memory_search_ranker_numeric": joblib.load(out_dir / "no_memory_search_ranker_numeric.joblib"),
        "numeric_memory_search_ranker": joblib.load(out_dir / "numeric_memory_search_ranker.joblib"),
    }
    data = []
    data.extend(lm.load_split(args.hotpot_dev, args.eval_per_dataset, "hotpotqa"))
    data.extend(lm.load_split(args.twiki_dev, args.eval_per_dataset, "2wikimultihopqa"))
    cache_path = out_dir / "numeric_memory_online_cache.pkl"
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() and not args.rebuild_cache else {}
    session = requests.Session()
    tokenizer, answer_model = lm.load_answer_model(args.model)
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": vars(args), "methods": {}}
    for method in ["heuristic_2search", "no_memory_search_ranker_numeric", "numeric_memory_search_ranker"]:
        rows = []
        bundle = bundles.get(method)
        print("ONLINE_NUMERIC", method, flush=True)
        for i, item in enumerate(data):
            row = run_item(item, method, bundle, session, cache, args, tokenizer, answer_model)
            rows.append(row)
            print(method, i, item.get("dataset"), item.get("id"), "searches", row["searches"], "cov", row["support_coverage"], "hit", row["final_answer_hit"], "first", row["first_support_hit_turn"], "nohit", row["no_hit_searches"], flush=True)
            if (i + 1) % 10 == 0:
                cache_path.write_bytes(pickle.dumps(cache))
        result["methods"][method] = {"summary": lm.summarize_online(rows), "rows": rows}
        (out_dir / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("ONLINE_NUMERIC_SUMMARY", method, json.dumps(result["methods"][method]["summary"], ensure_ascii=False), flush=True)
    del answer_model
    gc.collect()
    torch.cuda.empty_cache()
    cache_path.write_bytes(pickle.dumps(cache))
    result["summary"] = {k: v["summary"] for k, v in result["methods"].items()}
    (out_dir / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("ONLINE_NUMERIC_RESULT_PATH", str(out_dir / args.output), flush=True)
    print("ONLINE_NUMERIC_SUMMARY_JSON", json.dumps(result["summary"], ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    p.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki_n300")
    p.add_argument("--hotpot-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/dev.jsonl")
    p.add_argument("--twiki-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/dev.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--eval-per-dataset", type=int, default=25)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--min-searches", type=int, default=2)
    p.add_argument("--search-threshold", type=float, default=0.25)
    p.add_argument("--no-memory-c", type=float, default=2.0)
    p.add_argument("--memory-c", type=float, default=0.2)
    p.add_argument("--answer-max-new-tokens", type=int, default=48)
    p.add_argument("--output", default="numeric_memory_search_online_eval_n50.json")
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"), flush=True)
    print("SEARCH_R1_URL", lm.SEARCH_R1_URL, flush=True)
    if args.mode in {"train", "all"}:
        fit_and_save(args)
    if args.mode in {"eval", "all"}:
        eval_online(args)


if __name__ == "__main__":
    main()
