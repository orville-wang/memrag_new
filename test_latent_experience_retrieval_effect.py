import argparse
import gc
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MEMGEN_ROOT = "/root/autodl-tmp/memrag_new/MemGen"
sys.path.insert(0, MEMGEN_ROOT)
from memgen.model import MemGenModel  # noqa: E402

SEARCH_R1_URL = os.environ.get("SEARCH_R1_URL", "http://127.0.0.1:8000/retrieve")


def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_question(q):
    q = q.strip()
    if q and not q.endswith("?"):
        q += "?"
    return q


def clean_query(text, fallback=""):
    text = text.strip()
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"</?search>", "", text, flags=re.I).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    text = re.sub(r"^(search query|query|answer)\s*[:\-]\s*", "", text, flags=re.I).strip()
    text = text.strip(" `\"'")
    text = re.sub(r"\s+", " ", text)
    return text[:220] or fallback


def retrieve(session, query, topk=5):
    resp = session.post(
        SEARCH_R1_URL,
        json={"queries": [query], "topk": topk, "return_scores": True},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["result"][0]


def hit_info(hits, golden_answers):
    golds = [normalize_text(g) for g in golden_answers if str(g).strip()]
    for rank, hit in enumerate(hits, 1):
        doc = hit["document"]
        contents_norm = normalize_text(doc.get("contents", ""))
        for gold in golds:
            if gold and gold in contents_norm:
                return True, rank
    return False, None


def doc_title(doc):
    contents = doc.get("contents", "")
    return contents.split("\n", 1)[0].strip('"') if contents else ""


def doc_snippet(doc, n=220):
    contents = doc.get("contents", "")
    if "\n" in contents:
        contents = contents.split("\n", 1)[1]
    return contents[:n].replace("\n", " ")


def build_experiences(session, train_ds, need=3, scan=80, topk=5):
    experiences = []
    for idx in range(min(scan, len(train_ds))):
        item = train_ds[idx]
        q = clean_question(item["question"])
        gold = item["golden_answers"]
        hits = retrieve(session, q, topk=topk)
        ok, rank = hit_info(hits, gold)
        if not ok:
            continue
        hit = hits[rank - 1]
        experiences.append({
            "index": idx,
            "question": q,
            "query": q,
            "golden_answers": gold,
            "hit_rank": rank,
            "hit_doc_id": hit["document"].get("id"),
            "hit_title": doc_title(hit["document"]),
            "hit_snippet": doc_snippet(hit["document"]),
        })
        if len(experiences) >= need:
            break
    if len(experiences) < need:
        raise RuntimeError(f"Only found {len(experiences)} successful experiences in first {scan} train examples")
    return experiences


def make_prompt(question, experiences=None):
    if not experiences:
        return (
            "Write one concise Wikipedia search query for BM25 retrieval. "
            "Keep only the entities and relation needed to find the answer. "
            "Output only the query.\n"
            f"Question: {question}"
        )
    exp_text = []
    for i, exp in enumerate(experiences, 1):
        exp_text.append(
            f"Experience {i}:\n"
            f"Question: {exp['question']}\n"
            f"Successful query: {exp['query']}\n"
            f"Retrieved evidence title: {exp['hit_title']}\n"
            f"Evidence snippet: {exp['hit_snippet']}"
        )
    return (
        "Use the past successful retrieval experiences to write a better next search query. "
        "The query should be concise and optimized for Wikipedia BM25 retrieval. "
        "Output only the query.\n\n"
        + "\n\n".join(exp_text)
        + f"\n\nTarget question: {question}"
    )


def generate_base(model, tokenizer, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return clean_query(decoded, fallback="")


def generate_memgen(model, tokenizer, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_cfg.weaver_do_sample = False
    gen_cfg.trigger_do_sample = False
    gen_cfg.temperature = 0.0
    with torch.no_grad():
        output_ids, aug_mask = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=gen_cfg,
            return_augmentation_mask=True,
        )
    decoded = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return clean_query(decoded, fallback=""), aug_mask.detach().cpu().tolist()


def evaluate_query(session, query, gold, topk):
    hits = retrieve(session, query, topk=topk)
    ok, rank = hit_info(hits, gold)
    return {
        "query": query,
        "hit": ok,
        "rank": rank,
        "top1_score": hits[0]["score"] if hits else None,
        "top1_doc_id": hits[0]["document"].get("id") if hits else None,
        "top1_title": doc_title(hits[0]["document"]) if hits else None,
    }


def summarize(rows, methods):
    summary = {}
    n = len(rows)
    for method in methods:
        hits = [row["results"][method]["hit"] for row in rows]
        ranks = [row["results"][method]["rank"] for row in rows if row["results"][method]["rank"]]
        mrr = sum(1.0 / r for r in ranks) / n
        summary[method] = {
            "n": n,
            "hit_at_k": sum(hits),
            "hit_rate": sum(hits) / n if n else 0.0,
            "mrr_at_k": mrr,
        }
    return summary


def free_cuda(*objs):
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--memgen-load-model-path", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model")
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", default="/root/autodl-tmp/memrag_new/latent_experience_retrieval_ab_test.json")
    args = parser.parse_args()

    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
    print("MODEL", args.model)
    print("SEARCH_R1_URL", SEARCH_R1_URL)
    print("CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

    session = requests.Session()
    nq = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")
    train_ds = nq["train"]
    test_ds = nq["test"].select(range(args.num_test))

    print("Building real retrieval experiences from official NQ train...")
    experiences = build_experiences(session, train_ds, need=3, scan=80, topk=args.topk)
    for exp in experiences:
        print("EXPERIENCE", exp["index"], exp["question"], "=>", exp["hit_title"], "rank", exp["hit_rank"])

    methods = ["raw_question", "base_no_experience", "base_text_experience", "memgen_latent_experience"]
    rows = []
    for idx, item in enumerate(test_ds):
        q = clean_question(item["question"])
        gold = item["golden_answers"]
        rows.append({"index": idx, "question": q, "golden_answers": gold, "results": {}})
        rows[-1]["results"]["raw_question"] = evaluate_query(session, q, gold, args.topk)

    print("Loading base query generator...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()

    for row in rows:
        q = row["question"]
        gold = row["golden_answers"]
        query_a = generate_base(base_model, tokenizer, make_prompt(q, None), args.max_new_tokens)
        query_b = generate_base(base_model, tokenizer, make_prompt(q, experiences), args.max_new_tokens)
        row["results"]["base_no_experience"] = evaluate_query(session, query_a, gold, args.topk)
        row["results"]["base_text_experience"] = evaluate_query(session, query_b, gold, args.topk)
        print("BASE", row["index"], "raw_hit", row["results"]["raw_question"]["hit"], "no_exp", query_a, row["results"]["base_no_experience"]["hit"], "text_exp", query_b, row["results"]["base_text_experience"]["hit"])

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading trained MemGen checkpoint...")
    print("MEMGEN_LOAD_MODEL_PATH", args.memgen_load_model_path)
    lora = {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    memgen_config = {
        "model_name": args.model,
        "load_model_path": args.memgen_load_model_path,
        "attn_implementation": "eager",
        "torch_dtype": "bfloat16",
        "max_prompt_aug_num": 8,
        "max_inference_aug_num": 0,
        "weaver": {"model_name": args.model, "prompt_latents_len": 8, "inference_latents_len": 8, "lora_config": lora},
        "trigger": {"model_name": args.model, "active": False, "lora_config": lora},
    }
    memgen_model = MemGenModel.from_config(memgen_config).to(torch.bfloat16).cuda().eval()
    memgen_tok = memgen_model.tokenizer

    for row in rows:
        q = row["question"]
        gold = row["golden_answers"]
        query_c, aug_mask = generate_memgen(memgen_model, memgen_tok, make_prompt(q, experiences), args.max_new_tokens)
        row["results"]["memgen_latent_experience"] = evaluate_query(session, query_c, gold, args.topk)
        row["results"]["memgen_latent_experience"]["augmentation_mask"] = aug_mask
        print("MEMGEN", row["index"], query_c, row["results"]["memgen_latent_experience"]["hit"], "aug", aug_mask[:1])

    summary = summarize(rows, methods)
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "RUC-NLPIR/FlashRAG_datasets/nq",
        "retriever": "Search-R1 BM25 over official PeterJinGo/wiki-18 corpus",
        "model": args.model,
        "num_test": args.num_test,
        "topk": args.topk,
        "experience_source": "official NQ train examples whose raw question retrieval hit a golden answer",
        "experiences": experiences,
        "summary": summary,
        "rows": rows,
        "memgen_load_model_path": args.memgen_load_model_path,
        "caveat": "MemGen uses the released TriviaQA weaver-sft checkpoint. This checkpoint is trained for answer generation with prompt latent augmentation, not specifically for standalone search-query rewriting.",
    }
    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("SUMMARY_JSON", json.dumps(summary, ensure_ascii=False))
    print("RESULT_PATH", str(out))


if __name__ == "__main__":
    main()
