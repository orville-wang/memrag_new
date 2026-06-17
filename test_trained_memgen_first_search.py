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
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MEMGEN_ROOT = "/root/autodl-tmp/memrag_new/MemGen"
sys.path.insert(0, MEMGEN_ROOT)
from memgen.model import MemGenModel  # noqa: E402

SEARCH_R1_URL = os.environ.get("SEARCH_R1_URL", "http://127.0.0.1:8000/retrieve")
TRIVIAQA_SYSTEM_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""


def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def retrieve(session, query, topk):
    resp = session.post(SEARCH_R1_URL, json={"queries": [query], "topk": topk, "return_scores": True}, timeout=180)
    resp.raise_for_status()
    return resp.json()["result"][0]


def hit_info(hits, aliases):
    golds = [normalize_text(x) for x in aliases if str(x).strip()]
    for rank, hit in enumerate(hits, 1):
        contents = normalize_text(hit["document"].get("contents", ""))
        if any(g and g in contents for g in golds):
            return True, rank
    return False, None


def parse_action(text):
    search = re.search(r"<search>(.*?)</search>", text, flags=re.S | re.I)
    answer = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if search:
        q = search.group(1).strip().splitlines()[0].strip()
        return "search", re.sub(r"\s+", " ", q)[:240]
    if answer:
        return "answer", answer.group(1).strip().splitlines()[0].strip()[:240]
    return "other", text.strip().splitlines()[0][:240] if text.strip() else ""


def make_messages(question):
    return [
        {"role": "system", "content": TRIVIAQA_SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def generate_base(model, tokenizer, question, max_new_tokens):
    text = tokenizer.apply_chat_template(make_messages(question), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def generate_memgen(model, tokenizer, question, max_new_tokens):
    text = tokenizer.apply_chat_template(make_messages(question), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    gen_cfg.weaver_do_sample = False
    gen_cfg.trigger_do_sample = False
    gen_cfg.temperature = 0.0
    with torch.no_grad():
        output_ids, aug_mask = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], generation_config=gen_cfg, return_augmentation_mask=True)
    decoded = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return decoded, aug_mask.detach().cpu().tolist()


def eval_action(session, action_type, action_text, aliases, topk):
    result = {"action_type": action_type, "action_text": action_text, "hit": False, "rank": None, "top1_title": None}
    if action_type != "search" or not action_text:
        return result
    hits = retrieve(session, action_text, topk)
    ok, rank = hit_info(hits, aliases)
    result.update({"hit": ok, "rank": rank})
    if hits:
        contents = hits[0]["document"].get("contents", "")
        result["top1_title"] = contents.split("\n", 1)[0].strip('"')
    return result


def summarize(rows, key):
    n = len(rows)
    search_count = sum(1 for r in rows if r[key]["action_type"] == "search")
    hits = sum(1 for r in rows if r[key]["hit"])
    mrr = sum((1.0 / r[key]["rank"]) for r in rows if r[key]["rank"]) / n
    return {"n": n, "search_actions": search_count, "search_rate": search_count / n if n else 0, "hit_at_k": hits, "hit_rate": hits / n if n else 0, "mrr_at_k": mrr}


def load_trained_memgen(args):
    lora = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM"}
    cfg = {
        "model_name": args.model,
        "load_model_path": args.memgen_load_model_path,
        "attn_implementation": "eager",
        "torch_dtype": "bfloat16",
        "max_prompt_aug_num": 8,
        "max_inference_aug_num": 0,
        "weaver": {"model_name": args.model, "prompt_latents_len": 8, "inference_latents_len": 8, "lora_config": lora},
        "trigger": {"model_name": args.model, "active": False, "lora_config": lora},
    }
    model = MemGenModel.from_config(cfg).to(torch.bfloat16).cuda().eval()
    adapter = load_file(str(Path(args.memgen_load_model_path) / "weaver/weaver/adapter_model.safetensors"))
    ref_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    diff = None
    for n, p in model.weaver.model.named_parameters():
        if "layers.0.self_attn.q_proj.lora_A" in n:
            diff = torch.max(torch.abs(p.detach().cpu().float() - adapter[ref_key].float())).item()
            break
    return model, diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--memgen-load-model-path", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model")
    parser.add_argument("--num-test", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", default="/root/autodl-tmp/memrag_new/trained_memgen_first_search_n20.json")
    args = parser.parse_args()

    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
    print("MODEL", args.model)
    print("MEMGEN_LOAD_MODEL_PATH", args.memgen_load_model_path)
    print("SEARCH_R1_URL", SEARCH_R1_URL)

    session = requests.Session()
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split="validation").select(range(args.num_test))
    rows = []

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    print("Running vanilla base first-search actions...")
    for i, item in enumerate(ds):
        question = item["question"]
        aliases = item["answer"]["normalized_aliases"]
        out = generate_base(base, tokenizer, question, args.max_new_tokens)
        action_type, action_text = parse_action(out)
        row = {"index": i, "question": question, "golden_aliases": aliases, "base_output": out, "base": eval_action(session, action_type, action_text, aliases, args.topk)}
        rows.append(row)
        print("BASE", i, action_type, action_text, row["base"]["hit"])

    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("Running trained MemGen first-search actions...")
    memgen, lora_diff = load_trained_memgen(args)
    mtok = memgen.tokenizer
    print("LORA_COMPARE_MAX_ABS_DIFF", lora_diff)
    for row in rows:
        out, aug_mask = generate_memgen(memgen, mtok, row["question"], args.max_new_tokens)
        action_type, action_text = parse_action(out)
        row["memgen_output"] = out
        row["memgen_aug_mask"] = aug_mask
        row["memgen"] = eval_action(session, action_type, action_text, row["golden_aliases"], args.topk)
        print("MEMGEN", row["index"], action_type, action_text, row["memgen"]["hit"], "aug", aug_mask[:1])

    summary = {"base": summarize(rows, "base"), "trained_memgen": summarize(rows, "memgen")}
    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "mandarjoshi/trivia_qa/rc.wikipedia.nocontext validation",
        "retriever": "Search-R1 BM25 over official PeterJinGo/wiki-18 corpus",
        "model": args.model,
        "memgen_load_model_path": args.memgen_load_model_path,
        "lora_compare_max_abs_diff": lora_diff,
        "summary": summary,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("SUMMARY_JSON", json.dumps(summary, ensure_ascii=False))
    print("RESULT_PATH", args.output)


if __name__ == "__main__":
    main()
