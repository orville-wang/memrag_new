#!/usr/bin/env python3
import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch

import latent_memory_search_controller as lm
import dense_latent_action_scorer as dlas


def zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    if std < 1e-6:
        return np.zeros_like(x)
    return (x - float(x.mean())) / std


def memory_prior_scores(state_vec: np.ndarray, query_vecs: np.ndarray, memory_bank: Dict, topk: int) -> np.ndarray:
    mem_vec = dlas.memory_embedding(state_vec, memory_bank, exclude_state_id=None, topk=topk)
    return query_vecs @ mem_vec


def base_scores(model, state_vec: np.ndarray, query_vecs: np.ndarray) -> np.ndarray:
    state = torch.from_numpy(np.repeat(state_vec[None, :], len(query_vecs), axis=0)).cuda()
    query = torch.from_numpy(query_vecs).cuda()
    mem = torch.zeros_like(query)
    model.eval()
    with torch.no_grad():
        return model(state, query, mem).detach().cpu().numpy()


def offline_eval(encoded: dlas.EncodedRows, model, memory_bank: Dict, alpha: float, topk: int) -> Dict:
    states = len(encoded.groups)
    pos_states = top1_pos = top1_nohit = 0
    for sid, idxs in encoded.groups.items():
        rows = [encoded.rows[i] for i in idxs]
        state_vec = encoded.state_emb[encoded.state_text_to_idx[rows[0]["state_text"]]]
        query_vecs = np.stack([encoded.query_emb[encoded.query_text_to_idx[r["action_text"]]] for r in rows], axis=0)
        b = base_scores(model, state_vec, query_vecs)
        p = memory_prior_scores(state_vec, query_vecs, memory_bank, topk)
        scores = zscore(b) + alpha * zscore(p)
        best = int(scores.argmax())
        if any(r.get("label") for r in rows):
            pos_states += 1
            if rows[best].get("label"):
                top1_pos += 1
        if rows[best].get("no_hit"):
            top1_nohit += 1
    return {
        "states": states,
        "states_with_positive_search": pos_states,
        "top1_positive_rate_on_positive_states": top1_pos / pos_states if pos_states else 0.0,
        "top1_no_hit_rate": top1_nohit / states if states else 0.0,
    }


def sweep_alpha(encoded: dlas.EncodedRows, model, memory_bank: Dict, topk: int) -> List[Dict]:
    out = []
    for alpha in [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        ev = offline_eval(encoded, model, memory_bank, alpha, topk)
        ev["alpha"] = alpha
        ev["objective"] = ev["top1_positive_rate_on_positive_states"] - 0.4 * ev["top1_no_hit_rate"]
        out.append(ev)
    return sorted(out, key=lambda x: x["objective"], reverse=True)


def score_online(model, encoder, memory_bank, state_text: str, candidates: List[str], alpha: float, topk: int) -> List[Dict]:
    state_vec = encoder.encode([state_text], prefix="Represent this retrieval state: ")[0]
    query_vecs = encoder.encode(candidates, prefix="Represent this search query: ")
    b = base_scores(model, state_vec, query_vecs)
    p = memory_prior_scores(state_vec, query_vecs, memory_bank, topk)
    scores = zscore(b) + alpha * zscore(p)
    order = np.argsort(-scores)
    return [
        {
            "action_text": candidates[int(i)],
            "score": float(scores[int(i)]),
            "base_score": float(b[int(i)]),
            "memory_prior": float(p[int(i)]),
        }
        for i in order
    ]


def choose_heuristic(question: str, previous_queries: List[str], state_docs: List[Dict]) -> Tuple[str, List[Dict]]:
    candidates = lm.runtime_candidates(question, previous_queries, state_docs, max_candidates=24)
    prev = {lm.norm_key(q) for q in previous_queries}
    for cand in candidates:
        if lm.norm_key(cand) not in prev:
            return cand, []
    return candidates[0] if candidates else question, []


def run_item(method, model, encoder, memory_bank, item, session, cache, args, ans_tok, ans_model):
    support = lm.supporting_titles(item)
    answers = lm.golden_answers(item)
    previous_queries, state_docs, trajectory = [], [], []
    retrieved_support = set()
    repeated = nohit = 0
    first_support = first_answer = all_support = None
    finish_reason = "max_turn"
    for turn in range(1, args.max_turns + 1):
        if turn > args.min_searches and args.oracle_stop and support and len(retrieved_support) >= len({lm.norm_key(t) for t in support}):
            finish_reason = "answer"
            trajectory.append({"turn": turn, "action_type": "answer", "action_text": lm.ANSWER_ACTION, "stop_rule": "oracle_all_support"})
            break
        state_text = lm.state_text(item["question"], previous_queries, state_docs, turn - 1)
        candidates = lm.runtime_candidates(item["question"], previous_queries, state_docs, max_candidates=24) or [item["question"]]
        prev = {lm.norm_key(q) for q in previous_queries}
        if method == "heuristic":
            action_text, ranked = choose_heuristic(item["question"], previous_queries, state_docs)
        else:
            ranked = score_online(model, encoder, memory_bank, state_text, candidates, args.alpha if method == "late_fusion_latent" else 0.0, args.memory_topk)
            action_text = ranked[0]["action_text"]
            for cand in ranked:
                if lm.norm_key(cand["action_text"]) not in prev:
                    action_text = cand["action_text"]
                    break
        event = {"turn": turn, "action_type": "search", "action_text": action_text, "top_scored_actions": ranked[:8]}
        event["duplicate_query"] = lm.norm_key(action_text) in prev
        if event["duplicate_query"]:
            repeated += 1
        previous_queries.append(action_text)
        docs = lm.retrieve(session, action_text, args.topk, cache)
        hits = lm.doc_hits(docs, support, answers)
        for title in hits["hit_support_titles"]:
            retrieved_support.add(lm.norm_key(title))
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
    final_answer = lm.generate_answer(ans_tok, ans_model, item["question"], state_docs, args.answer_max_new_tokens)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki_n300")
    parser.add_argument("--encoder-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--answer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--eval-per-dataset", type=int, default=25)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--min-searches", type=int, default=2)
    parser.add_argument("--memory-topk", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--oracle-stop", action="store_true", default=True)
    parser.add_argument("--answer-max-new-tokens", type=int, default=48)
    parser.add_argument("--hotpot-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/dev.jsonl")
    parser.add_argument("--twiki-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/dev.jsonl")
    parser.add_argument("--output", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki_n300/dense_late_fusion_prior_online_eval_n50.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    encoder = dlas.DenseEncoder(args.encoder_model)
    train_rows = [r for r in lm.read_jsonl(out_dir / "train_state_actions.jsonl") if r.get("action_type") == "search"]
    dev_rows = [r for r in lm.read_jsonl(out_dir / "dev_state_actions.jsonl") if r.get("action_type") == "search"]
    train = dlas.encode_rows(train_rows, encoder, out_dir / "dense_latent_cache", "train_search")
    dev = dlas.encode_rows(dev_rows, encoder, out_dir / "dense_latent_cache", "dev_search")
    memory_bank = dlas.build_memory_bank(train)
    ckpt = torch.load(out_dir / "dense_latent_action_scorers.pt", map_location="cpu")
    model = dlas.ActionScorer(train.state_emb.shape[1], hidden=ckpt["settings"].get("hidden", 384)).cuda()
    model.load_state_dict(ckpt["dense_no_memory"])
    sweep = sweep_alpha(dev, model, memory_bank, args.memory_topk)
    (out_dir / "dense_late_fusion_alpha_sweep.json").write_text(json.dumps(sweep, ensure_ascii=False, indent=2))
    print("ALPHA_SWEEP_TOP", json.dumps(sweep[:6], ensure_ascii=False), flush=True)
    best_alpha = sweep[0]["alpha"]
    if args.alpha == 999:
        args.alpha = best_alpha
    print("USING_ALPHA", args.alpha, flush=True)

    data = []
    data.extend(lm.load_split(args.hotpot_dev, args.eval_per_dataset, "hotpotqa"))
    data.extend(lm.load_split(args.twiki_dev, args.eval_per_dataset, "2wikimultihopqa"))
    cache_path = out_dir / "dense_late_fusion_online_cache.pkl"
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    session = requests.Session()
    ans_tok, ans_model = dlas.load_answer_model(args.answer_model)
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": vars(args), "alpha_sweep_top": sweep[:6], "methods": {}}
    for method in ["heuristic", "dense_no_memory", "late_fusion_latent"]:
        rows = []
        print("ONLINE_LATE_FUSION", method, flush=True)
        for i, item in enumerate(data):
            row = run_item(method, model, encoder, memory_bank, item, session, cache, args, ans_tok, ans_model)
            rows.append(row)
            print(method, i, item.get("dataset"), item.get("id"), "searches", row["searches"], "cov", row["support_coverage"], "hit", row["final_answer_hit"], "first", row["first_support_hit_turn"], "nohit", row["no_hit_searches"], flush=True)
            if (i + 1) % 10 == 0:
                cache_path.write_bytes(pickle.dumps(cache))
        result["methods"][method] = {"summary": lm.summarize_online(rows), "rows": rows}
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("ONLINE_LATE_FUSION_SUMMARY", method, json.dumps(result["methods"][method]["summary"], ensure_ascii=False), flush=True)
    del ans_model
    gc.collect()
    torch.cuda.empty_cache()
    result["summary"] = {k: v["summary"] for k, v in result["methods"].items()}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("ONLINE_LATE_FUSION_RESULT_PATH", args.output, flush=True)
    print("ONLINE_LATE_FUSION_SUMMARY_JSON", json.dumps(result["summary"], ensure_ascii=False), flush=True)
    encoder.close()


if __name__ == "__main__":
    main()
