#!/usr/bin/env python3
import argparse
import gc
import json
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import latent_memory_search_controller as lm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def group_by_state(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(row["state_id"], []).append(row)
    return groups


class DenseEncoder:
    def __init__(self, model_name: str, max_length: int = 384, batch_size: int = 64):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()

    def encode(self, texts: List[str], prefix: str = "") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        vecs = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[start : start + self.batch_size]]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model(**tok).last_hidden_state[:, 0]
                out = F.normalize(out.float(), p=2, dim=-1)
            vecs.append(out.cpu().numpy().astype(np.float32))
        return np.concatenate(vecs, axis=0)

    def close(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def unique_texts(rows: List[Dict], field: str) -> List[str]:
    seen, out = set(), []
    for row in rows:
        text = row[field]
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


@dataclass
class EncodedRows:
    rows: List[Dict]
    groups: Dict[str, List[int]]
    state_emb: np.ndarray
    query_emb: np.ndarray
    state_text_to_idx: Dict[str, int]
    query_text_to_idx: Dict[str, int]


def encode_rows(rows: List[Dict], encoder: DenseEncoder, cache_dir: Path, split_name: str) -> EncodedRows:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split_name}_dense_cache.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    state_texts = unique_texts(rows, "state_text")
    query_texts = unique_texts(rows, "action_text")
    print(f"ENCODE {split_name}: states={len(state_texts)} queries={len(query_texts)}", flush=True)
    state_emb = encoder.encode(state_texts, prefix="Represent this retrieval state: ")
    query_emb = encoder.encode(query_texts, prefix="Represent this search query: ")
    state_text_to_idx = {t: i for i, t in enumerate(state_texts)}
    query_text_to_idx = {t: i for i, t in enumerate(query_texts)}
    groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(row["state_id"], []).append(idx)
    data = EncodedRows(rows, groups, state_emb, query_emb, state_text_to_idx, query_text_to_idx)
    cache_path.write_bytes(pickle.dumps(data))
    return data


def build_memory_bank(encoded: EncodedRows) -> Dict:
    state_vecs, action_vecs, state_ids, records = [], [], [], []
    for row in encoded.rows:
        if not row.get("label"):
            continue
        state_vecs.append(encoded.state_emb[encoded.state_text_to_idx[row["state_text"]]])
        action_vecs.append(encoded.query_emb[encoded.query_text_to_idx[row["action_text"]]])
        state_ids.append(row["state_id"])
        records.append({
            "state_id": row["state_id"],
            "question": row["question"],
            "action_text": row["action_text"],
            "outcome_score": float(row.get("outcome_score", 0.0)),
        })
    return {
        "state_emb": np.asarray(state_vecs, dtype=np.float32),
        "action_emb": np.asarray(action_vecs, dtype=np.float32),
        "state_ids": state_ids,
        "records": records,
    }


def memory_embedding(
    state_vec: np.ndarray,
    memory_bank: Dict,
    exclude_state_id: Optional[str] = None,
    topk: int = 8,
    temperature: float = 0.07,
) -> np.ndarray:
    mem_states = memory_bank["state_emb"]
    if mem_states.size == 0:
        return np.zeros_like(state_vec)
    sims = mem_states @ state_vec
    if exclude_state_id is not None:
        for idx, sid in enumerate(memory_bank["state_ids"]):
            if sid == exclude_state_id:
                sims[idx] = -1e9
    k = min(topk, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    weights = np.exp((sims[top] - sims[top].max()) / temperature)
    weights = weights / max(weights.sum(), 1e-12)
    return (weights[:, None] * memory_bank["action_emb"][top]).sum(axis=0).astype(np.float32)


class ActionScorer(nn.Module):
    def __init__(self, dim: int, hidden: int = 384):
        super().__init__()
        in_dim = dim * 6
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, state: torch.Tensor, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([state, query, memory, state * query, query * memory, torch.abs(query - memory)], dim=-1)
        return self.net(feat).squeeze(-1)


def group_tensors(
    encoded: EncodedRows,
    state_id: str,
    memory_bank: Dict,
    use_memory: bool,
    train_mode: bool,
    memory_topk: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    idxs = encoded.groups[state_id]
    rows = [encoded.rows[i] for i in idxs]
    state_vec = encoded.state_emb[encoded.state_text_to_idx[rows[0]["state_text"]]]
    if use_memory:
        mem_vec = memory_embedding(
            state_vec,
            memory_bank,
            exclude_state_id=state_id if train_mode else None,
            topk=memory_topk,
        )
    else:
        mem_vec = np.zeros_like(state_vec)
    q_vecs = np.stack([encoded.query_emb[encoded.query_text_to_idx[r["action_text"]]] for r in rows], axis=0)
    labels = np.asarray([1 if r.get("label") else 0 for r in rows], dtype=np.int64)
    pos = np.flatnonzero(labels)
    target = int(pos[0]) if len(pos) else 0
    state = torch.from_numpy(np.repeat(state_vec[None, :], len(rows), axis=0)).to(device)
    query = torch.from_numpy(q_vecs).to(device)
    mem = torch.from_numpy(np.repeat(mem_vec[None, :], len(rows), axis=0)).to(device)
    target_t = torch.tensor([target], dtype=torch.long, device=device)
    return state, query, mem, target_t, rows


def offline_eval(
    model: ActionScorer,
    encoded: EncodedRows,
    memory_bank: Dict,
    use_memory: bool,
    memory_topk: int,
    device: str,
) -> Dict:
    model.eval()
    states = len(encoded.groups)
    pos_states = top1_pos = top1_nohit = 0
    with torch.no_grad():
        for sid, idxs in encoded.groups.items():
            state, query, mem, _, rows = group_tensors(encoded, sid, memory_bank, use_memory, False, memory_topk, device)
            scores = model(state, query, mem).detach().cpu().numpy()
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


def train_scorer(
    train: EncodedRows,
    dev: EncodedRows,
    memory_bank: Dict,
    use_memory: bool,
    args,
    name: str,
) -> Tuple[ActionScorer, Dict]:
    device = "cuda"
    dim = train.state_emb.shape[1]
    model = ActionScorer(dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    state_ids = list(train.groups.keys())
    best_summary, best_state = None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(state_ids)
        total_loss = 0.0
        n = 0
        for sid in state_ids:
            rows = [train.rows[i] for i in train.groups[sid]]
            if not any(r.get("label") for r in rows):
                continue
            state, query, mem, target, _ = group_tensors(train, sid, memory_bank, use_memory, True, args.memory_topk, device)
            scores = model(state, query, mem).unsqueeze(0)
            loss = F.cross_entropy(scores, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n += 1
        train_eval = offline_eval(model, train, memory_bank, use_memory, args.memory_topk, device)
        dev_eval = offline_eval(model, dev, memory_bank, use_memory, args.memory_topk, device)
        summary = {"epoch": epoch, "loss": total_loss / max(n, 1), "train": train_eval, "dev": dev_eval}
        print("DENSE_EPOCH", name, json.dumps(summary, ensure_ascii=False), flush=True)
        score = dev_eval["top1_positive_rate_on_positive_states"] - 0.4 * dev_eval["top1_no_hit_rate"]
        best_score = -1e9 if best_summary is None else best_summary["score"]
        if score > best_score:
            best_summary = {**summary, "score": score}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_summary


def choose_heuristic(question: str, previous_queries: List[str], state_docs: List[Dict]) -> Tuple[str, List[Dict]]:
    candidates = lm.runtime_candidates(question, previous_queries, state_docs, max_candidates=24)
    prev = {lm.norm_key(q) for q in previous_queries}
    for cand in candidates:
        if lm.norm_key(cand) not in prev:
            return cand, []
    return candidates[0] if candidates else question, []


def score_candidates_online(
    model: ActionScorer,
    encoder: DenseEncoder,
    memory_bank: Dict,
    state_text: str,
    candidates: List[str],
    use_memory: bool,
    args,
) -> List[Dict]:
    state_vec = encoder.encode([state_text], prefix="Represent this retrieval state: ")[0]
    query_vecs = encoder.encode(candidates, prefix="Represent this search query: ")
    mem_vec = memory_embedding(state_vec, memory_bank, exclude_state_id=None, topk=args.memory_topk) if use_memory else np.zeros_like(state_vec)
    state = torch.from_numpy(np.repeat(state_vec[None, :], len(candidates), axis=0)).cuda()
    query = torch.from_numpy(query_vecs).cuda()
    mem = torch.from_numpy(np.repeat(mem_vec[None, :], len(candidates), axis=0)).cuda()
    model.eval()
    with torch.no_grad():
        scores = model(state, query, mem).detach().cpu().numpy()
    order = np.argsort(-scores)
    return [{"action_text": candidates[int(i)], "score": float(scores[int(i)])} for i in order]


def load_answer_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    return tok, model


def run_item(method: str, model: Optional[ActionScorer], encoder: DenseEncoder, memory_bank: Dict, item: Dict, session, cache, args, ans_tok, ans_model) -> Dict:
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
        candidates = lm.runtime_candidates(item["question"], previous_queries, state_docs, max_candidates=24)
        if not candidates:
            candidates = [item["question"]]
        prev = {lm.norm_key(q) for q in previous_queries}
        if method == "heuristic":
            action_text, ranked = choose_heuristic(item["question"], previous_queries, state_docs)
        else:
            ranked = score_candidates_online(model, encoder, memory_bank, state_text, candidates, method == "dense_latent_memory", args)
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


def eval_online(models: Dict[str, ActionScorer], encoder: DenseEncoder, memory_bank: Dict, args) -> Dict:
    data = []
    data.extend(lm.load_split(args.hotpot_dev, args.eval_per_dataset, "hotpotqa"))
    data.extend(lm.load_split(args.twiki_dev, args.eval_per_dataset, "2wikimultihopqa"))
    cache_path = Path(args.output_dir) / "dense_latent_online_cache.pkl"
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() and not args.rebuild_cache else {}
    session = requests.Session()
    ans_tok, ans_model = load_answer_model(args.answer_model)
    result = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": vars(args), "methods": {}}
    for method in ["heuristic", "dense_no_memory", "dense_latent_memory"]:
        rows = []
        print("ONLINE_DENSE", method, flush=True)
        for i, item in enumerate(data):
            row = run_item(method, models.get(method), encoder, memory_bank, item, session, cache, args, ans_tok, ans_model)
            rows.append(row)
            print(method, i, item.get("dataset"), item.get("id"), "searches", row["searches"], "cov", row["support_coverage"], "hit", row["final_answer_hit"], "first", row["first_support_hit_turn"], "nohit", row["no_hit_searches"], flush=True)
            if (i + 1) % 10 == 0:
                cache_path.write_bytes(pickle.dumps(cache))
        result["methods"][method] = {"summary": lm.summarize_online(rows), "rows": rows}
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("ONLINE_DENSE_SUMMARY", method, json.dumps(result["methods"][method]["summary"], ensure_ascii=False), flush=True)
    del ans_model
    gc.collect()
    torch.cuda.empty_cache()
    cache_path.write_bytes(pickle.dumps(cache))
    result["summary"] = {k: v["summary"] for k, v in result["methods"].items()}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("ONLINE_DENSE_RESULT_PATH", args.output, flush=True)
    print("ONLINE_DENSE_SUMMARY_JSON", json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki_n300")
    parser.add_argument("--train-actions", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki_n300/train_state_actions.jsonl")
    parser.add_argument("--dev-actions", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki_n300/dev_state_actions.jsonl")
    parser.add_argument("--hotpot-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/dev.jsonl")
    parser.add_argument("--twiki-dev", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/dev.jsonl")
    parser.add_argument("--encoder-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--answer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--memory-topk", type=int, default=8)
    parser.add_argument("--eval-per-dataset", type=int, default=25)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--min-searches", type=int, default=2)
    parser.add_argument("--oracle-stop", action="store_true", default=True)
    parser.add_argument("--answer-max-new-tokens", type=int, default=48)
    parser.add_argument("--output", default="/root/autodl-tmp/memrag_new/latent_controller_hotpot2wiki_n300/dense_latent_action_online_eval_n50.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"), flush=True)
    print("SEARCH_R1_URL", lm.SEARCH_R1_URL, flush=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_rows = [r for r in lm.read_jsonl(Path(args.train_actions)) if r.get("action_type") == "search"]
    dev_rows = [r for r in lm.read_jsonl(Path(args.dev_actions)) if r.get("action_type") == "search"]
    encoder = DenseEncoder(args.encoder_model)
    cache_dir = Path(args.output_dir) / "dense_latent_cache"
    train = encode_rows(train_rows, encoder, cache_dir, "train_search")
    dev = encode_rows(dev_rows, encoder, cache_dir, "dev_search")
    memory_bank = build_memory_bank(train)
    print("MEMORY_BANK", memory_bank["state_emb"].shape, flush=True)

    models, summaries = {}, {}
    no_mem, no_mem_summary = train_scorer(train, dev, memory_bank, False, args, "dense_no_memory")
    lat_mem, lat_mem_summary = train_scorer(train, dev, memory_bank, True, args, "dense_latent_memory")
    models["dense_no_memory"] = no_mem
    models["dense_latent_memory"] = lat_mem
    summaries["dense_no_memory"] = no_mem_summary
    summaries["dense_latent_memory"] = lat_mem_summary
    (Path(args.output_dir) / "dense_latent_train_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    torch.save({"dense_no_memory": no_mem.state_dict(), "dense_latent_memory": lat_mem.state_dict(), "settings": vars(args)}, Path(args.output_dir) / "dense_latent_action_scorers.pt")
    print("DENSE_TRAIN_SUMMARY", json.dumps(summaries, ensure_ascii=False), flush=True)
    eval_online(models, encoder, memory_bank, args)
    encoder.close()


if __name__ == "__main__":
    main()
