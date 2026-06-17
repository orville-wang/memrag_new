#!/usr/bin/env python3
import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import requests
import torch

import latent_memory_search_controller as lm


def group_by_state(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups = {}
    for r in rows:
        groups.setdefault(r["state_id"], []).append(r)
    return groups


def offline_search_eval(bundle: lm.ControllerBundle, rows: List[Dict]) -> Dict:
    groups = group_by_state(rows)
    total = 0
    with_positive = 0
    top1_positive = 0
    top1_no_hit = 0
    avg_best_score = []
    for _, cand_rows in groups.items():
        if not cand_rows:
            continue
        probs = bundle.predict_proba(cand_rows)
        best_i = int(np.argmax(probs))
        best = cand_rows[best_i]
        total += 1
        avg_best_score.append(float(probs[best_i]))
        if any(r.get("label") for r in cand_rows):
            with_positive += 1
            if best.get("label"):
                top1_positive += 1
        if best.get("no_hit"):
            top1_no_hit += 1
    return {
        "states": total,
        "states_with_positive_search": with_positive,
        "top1_positive_rate_on_positive_states": top1_positive / with_positive if with_positive else 0.0,
        "top1_no_hit_rate": top1_no_hit / total if total else 0.0,
        "avg_best_search_score": sum(avg_best_score) / len(avg_best_score) if avg_best_score else None,
    }


def fit_and_save(args) -> Dict:
    out_dir = Path(args.output_dir)
    train_rows = [r for r in lm.read_jsonl(out_dir / "train_state_actions.jsonl") if r.get("action_type") == "search"]
    dev_rows = [r for r in lm.read_jsonl(out_dir / "dev_state_actions.jsonl") if r.get("action_type") == "search"]
    summaries = {}
    for name, use_memory in [("no_memory_search_ranker", False), ("latent_memory_search_ranker", True)]:
        print("TRAIN_SEARCH", name, "rows", len(train_rows), "use_memory", use_memory, flush=True)
        bundle = lm.fit_controller(train_rows, use_memory=use_memory)
        summaries[name] = {
            "train": offline_search_eval(bundle, train_rows),
            "dev": offline_search_eval(bundle, dev_rows),
        }
        joblib.dump(bundle, out_dir / f"{name}.joblib")
        print("SEARCH_OFFLINE", name, json.dumps(summaries[name], ensure_ascii=False), flush=True)
    (out_dir / "search_only_train_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    return summaries


def choose_search_action(bundle, item, turn, previous_queries, state_docs, threshold, min_searches):
    cand_queries = lm.runtime_candidates(item["question"], previous_queries, state_docs, max_candidates=20)
    prev_norm = {lm.norm_key(q) for q in previous_queries}
    # Keep candidates unique but allow fallback if all are repeats.
    candidates = [("search", q) for q in cand_queries]
    if not candidates:
        return "answer", lm.ANSWER_ACTION, 0.0, []
    rows = lm.make_runtime_action_rows(item, f"online_search:{item.get('dataset')}:{item.get('id')}:{turn}", turn - 1, previous_queries, state_docs, candidates)
    probs = bundle.predict_proba(rows)
    order = np.argsort(-probs)
    ranked = []
    chosen = None
    for idx in order:
        idx = int(idx)
        q = rows[idx]["action_text"]
        score = float(probs[idx])
        ranked.append({"action_type": "search", "action_text": q, "score": score})
        if chosen is None and lm.norm_key(q) not in prev_norm:
            chosen = (q, score)
    if chosen is None:
        idx = int(order[0])
        chosen = (rows[idx]["action_text"], float(probs[idx]))
    if turn > min_searches and chosen[1] < threshold:
        return "answer", lm.ANSWER_ACTION, chosen[1], ranked[:6]
    return "search", chosen[0], chosen[1], ranked[:6]


def run_item(item, method, bundle, session, cache, args, tokenizer, answer_model):
    support = lm.supporting_titles(item)
    answers = lm.golden_answers(item)
    previous_queries = []
    state_docs = []
    retrieved_support = set()
    trajectory = []
    repeated_queries = 0
    no_hit_searches = 0
    first_support_hit_turn = None
    first_answer_hit_turn = None
    all_support_hit_turn = None
    finish_reason = "max_turn"

    for turn in range(1, args.max_turns + 1):
        if method == "heuristic_2search":
            action_type, action_text, score, ranked = lm.choose_heuristic(item["question"], previous_queries, state_docs, turn)
        else:
            action_type, action_text, score, ranked = choose_search_action(bundle, item, turn, previous_queries, state_docs, args.search_threshold, args.min_searches)
        event = {"turn": turn, "action_type": action_type, "action_text": action_text, "controller_score": score, "top_scored_actions": ranked}
        if action_type == "answer":
            finish_reason = "answer"
            trajectory.append(event)
            break
        qn = lm.norm_key(action_text)
        event["duplicate_query"] = qn in {lm.norm_key(q) for q in previous_queries}
        if event["duplicate_query"]:
            repeated_queries += 1
        previous_queries.append(action_text)
        docs = lm.retrieve(session, action_text, args.topk, cache)
        hits = lm.doc_hits(docs, support, answers)
        for t in hits["hit_support_titles"]:
            retrieved_support.add(lm.norm_key(t))
        if hits["hit_support_titles"] and first_support_hit_turn is None:
            first_support_hit_turn = turn
        if hits["hit_answer"] and first_answer_hit_turn is None:
            first_answer_hit_turn = turn
        if support and len(retrieved_support) >= len({lm.norm_key(t) for t in support}) and all_support_hit_turn is None:
            all_support_hit_turn = turn
        if not hits["hit_support_titles"] and not hits["hit_answer"]:
            no_hit_searches += 1
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
        "repeated_queries": repeated_queries,
        "no_hit_searches": no_hit_searches,
        "support_coverage": len(retrieved_support) / len({lm.norm_key(t) for t in support}) if support else None,
        "any_support_hit": first_support_hit_turn is not None,
        "first_support_hit_turn": first_support_hit_turn,
        "first_answer_hit_turn": first_answer_hit_turn,
        "all_support_hit_turn": all_support_hit_turn,
        "trajectory": trajectory,
    }


def eval_online(args) -> Dict:
    out_dir = Path(args.output_dir)
    bundles = {
        "no_memory_search_ranker": joblib.load(out_dir / "no_memory_search_ranker.joblib"),
        "latent_memory_search_ranker": joblib.load(out_dir / "latent_memory_search_ranker.joblib"),
    }
    data = []
    data.extend(lm.load_split(args.hotpot_dev, args.eval_per_dataset, "hotpotqa"))
    data.extend(lm.load_split(args.twiki_dev, args.eval_per_dataset, "2wikimultihopqa"))
    cache_path = out_dir / "search_only_online_cache.pkl"
    if cache_path.exists() and not args.rebuild_cache:
        cache = pickle.loads(cache_path.read_bytes())
    else:
        cache = {}
    session = requests.Session()
    tokenizer, answer_model = lm.load_answer_model(args.model)
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": vars(args),
        "methods": {},
    }
    methods = ["heuristic_2search", "no_memory_search_ranker", "latent_memory_search_ranker"]
    for method in methods:
        rows = []
        bundle = bundles.get(method)
        print("ONLINE_SEARCH_ONLY", method, flush=True)
        for i, item in enumerate(data):
            row = run_item(item, method, bundle, session, cache, args, tokenizer, answer_model)
            rows.append(row)
            print(method, i, item.get("dataset"), item.get("id"), "searches", row["searches"], "cov", row["support_coverage"], "hit", row["final_answer_hit"], "first", row["first_support_hit_turn"], "nohit", row["no_hit_searches"], flush=True)
            if (i + 1) % 10 == 0:
                cache_path.write_bytes(pickle.dumps(cache))
        result["methods"][method] = {"summary": lm.summarize_online(rows), "rows": rows}
        (out_dir / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("ONLINE_SEARCH_SUMMARY", method, json.dumps(result["methods"][method]["summary"], ensure_ascii=False), flush=True)
    del answer_model
    gc.collect()
    torch.cuda.empty_cache()
    cache_path.write_bytes(pickle.dumps(cache))
    result["summary"] = {k: v["summary"] for k, v in result["methods"].items()}
    (out_dir / args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("ONLINE_SEARCH_RESULT_PATH", str(out_dir / args.output), flush=True)
    print("ONLINE_SEARCH_SUMMARY_JSON", json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki_n300")
    parser.add_argument("--hotpot-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/dev.jsonl")
    parser.add_argument("--twiki-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/dev.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--eval-per-dataset", type=int, default=25)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--min-searches", type=int, default=1)
    parser.add_argument("--search-threshold", type=float, default=0.45)
    parser.add_argument("--answer-max-new-tokens", type=int, default=48)
    parser.add_argument("--output", default="latent_search_only_controller_online_eval_n50.json")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"), flush=True)
    print("SEARCH_R1_URL", lm.SEARCH_R1_URL, flush=True)
    if args.mode in {"train", "all"}:
        fit_and_save(args)
    if args.mode in {"eval", "all"}:
        eval_online(args)


if __name__ == "__main__":
    main()
