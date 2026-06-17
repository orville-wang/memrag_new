import argparse
import gc
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from flashrag.pipeline.reasoning_pipeline import SearchR1Pipeline

MEMGEN_ROOT = "/root/autodl-tmp/memrag_new/MemGen"
sys.path.insert(0, MEMGEN_ROOT)
from memgen.model import MemGenModel  # noqa: E402

SEARCH_R1_URL = os.environ.get("SEARCH_R1_URL", "http://127.0.0.1:8000/retrieve")
STOP_STRINGS = [
    "</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n",
    "</answer>", "</answer>\n", "</answer>\n\n", "<|endoftext|>", "<|im_end|>",
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_flashrag_jsonl(path: str, limit: int, dataset_name: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            item = json.loads(line)
            item["dataset"] = dataset_name
            rows.append(item)
    return rows


def supporting_titles(item: Dict) -> List[str]:
    sf = item.get("metadata", {}).get("supporting_facts", {})
    titles = sf.get("title", []) if isinstance(sf, dict) else []
    out = []
    seen = set()
    for title in titles:
        key = normalize_text(title)
        if key and key not in seen:
            out.append(title)
            seen.add(key)
    return out


def title_from_doc(doc: Dict) -> str:
    contents = doc.get("contents", "") or doc.get("document", {}).get("contents", "")
    title = contents.split("\n", 1)[0].strip().strip('"')
    return title


def contents_from_doc(doc: Dict) -> str:
    return doc.get("contents", "") or doc.get("document", {}).get("contents", "")


def retrieve(session: requests.Session, query: str, topk: int) -> List[Dict]:
    resp = session.post(SEARCH_R1_URL, json={"queries": [query], "topk": topk, "return_scores": True}, timeout=180)
    resp.raise_for_status()
    raw_hits = resp.json()["result"][0]
    docs = []
    for hit in raw_hits:
        d = hit.get("document", hit)
        docs.append({"id": d.get("id"), "contents": d.get("contents", ""), "score": hit.get("score")})
    return docs


def docs_to_information(docs: List[Dict], max_doc_chars: int) -> str:
    text = ""
    for i, doc in enumerate(docs, 1):
        contents = contents_from_doc(doc)
        title = contents.split("\n", 1)[0].strip().strip('"')
        body = contents.split("\n", 1)[1] if "\n" in contents else ""
        body = re.sub(r"\s+", " ", body).strip()[:max_doc_chars]
        text += f"Doc {i}(Title: {title}) {body}\n"
    return f"\n\n<information>\n{text}\n</information>\n\n"


def build_initial_prompt(tokenizer, question: str) -> str:
    user = SearchR1Pipeline.user_prompt.format(question=question)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return user


def truncate_prompt(tokenizer, prompt: str, max_input_tokens: int) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_input_tokens:
        return prompt
    head = max_input_tokens // 2
    tail = max_input_tokens - head
    return tokenizer.decode(ids[:head], skip_special_tokens=True) + tokenizer.decode(ids[-tail:], skip_special_tokens=True)


def truncate_at_first_stop(text: str) -> str:
    best = None
    for stop in STOP_STRINGS:
        idx = text.find(stop)
        if idx >= 0:
            end = idx + len(stop)
            if best is None or end < best:
                best = end
    return text[:best].strip() if best is not None else text.strip()


def parse_step_action(text: str) -> Tuple[str, str]:
    s = re.search(r"<search>(.*?)</search>", text, flags=re.S | re.I)
    a = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    candidates = []
    if s:
        candidates.append((s.start(), "search", re.sub(r"\s+", " ", s.group(1)).strip()))
    if a:
        candidates.append((a.start(), "answer", re.sub(r"\s+", " ", a.group(1)).strip()))
    if candidates:
        _, typ, val = sorted(candidates, key=lambda x: x[0])[0]
        return typ, val[:300]
    return "other", text.strip()[:300]


def doc_hits(docs: List[Dict], support_titles: List[str], golden_answers: List[str]) -> Dict:
    support_norm = {normalize_text(t): t for t in support_titles}
    answer_norms = [normalize_text(x) for x in golden_answers if str(x).strip()]
    hit_titles = []
    hit_answer = False
    top_titles = []
    for doc in docs:
        title = title_from_doc(doc)
        top_titles.append(title)
        nt = normalize_text(title)
        nc = normalize_text(contents_from_doc(doc))
        for key, raw in support_norm.items():
            if key and (nt == key or key in nt or nt in key):
                if raw not in hit_titles:
                    hit_titles.append(raw)
        if any(a and a in nc for a in answer_norms):
            hit_answer = True
    return {"hit_support_titles": hit_titles, "hit_answer": hit_answer, "top_titles": top_titles}


def final_answer_hit(answer: str, golden_answers: List[str]) -> bool:
    na = normalize_text(answer)
    for g in golden_answers:
        ng = normalize_text(g)
        if ng and (ng in na or na in ng):
            return True
    return False


class BaseActor:
    name = "base_qwen"

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()

    def generate(self, prompt: str, max_new_tokens: int, max_input_tokens: int) -> Tuple[str, Optional[List]]:
        prompt = truncate_prompt(self.tokenizer, prompt, max_input_tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return truncate_at_first_stop(text), None

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class MemGenActor:
    name = "trained_memgen"

    def __init__(
        self,
        model_name: str,
        load_path: str,
        weaver_model_name: Optional[str] = None,
        trigger_model_name: Optional[str] = None,
        prompt_latents_len: int = 8,
        inference_latents_len: int = 8,
        max_prompt_aug_num: int = 8,
        max_inference_aug_num: int = 0,
    ):
        self.model = self._load(
            model_name,
            load_path,
            weaver_model_name=weaver_model_name,
            trigger_model_name=trigger_model_name,
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
        )
        self.tokenizer = self.model.tokenizer
        self.load_path = load_path
        self.model_name = model_name
        self.weaver_model_name = weaver_model_name or model_name
        self.trigger_model_name = trigger_model_name or model_name
        self.lora_diff = self._check_lora(load_path)

    def _load(
        self,
        model_name: str,
        load_path: str,
        weaver_model_name: Optional[str] = None,
        trigger_model_name: Optional[str] = None,
        prompt_latents_len: int = 8,
        inference_latents_len: int = 8,
        max_prompt_aug_num: int = 8,
        max_inference_aug_num: int = 0,
    ):
        weaver_model_name = weaver_model_name or model_name
        trigger_model_name = trigger_model_name or model_name
        lora = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.1, "bias": "none", "task_type": "CAUSAL_LM"}
        cfg = {
            "model_name": model_name,
            "load_model_path": load_path,
            "attn_implementation": "eager",
            "torch_dtype": "bfloat16",
            "max_prompt_aug_num": max_prompt_aug_num,
            "max_inference_aug_num": max_inference_aug_num,
            "weaver": {"model_name": weaver_model_name, "prompt_latents_len": prompt_latents_len, "inference_latents_len": inference_latents_len, "lora_config": lora},
            "trigger": {"model_name": trigger_model_name, "active": False, "lora_config": lora},
        }
        return MemGenModel.from_config(cfg).to(torch.bfloat16).cuda().eval()

    def _check_lora(self, load_path: str):
        adapter_path = Path(load_path) / "weaver/weaver/adapter_model.safetensors"
        if not adapter_path.exists():
            return None
        adapter = load_file(str(adapter_path))
        ref_key = next((k for k in adapter if "q_proj.lora_A.weight" in k), None)
        if ref_key is None:
            return None
        for n, p in self.model.weaver.model.named_parameters():
            if "q_proj.lora_A" in n and p.shape == adapter[ref_key].shape:
                return torch.max(torch.abs(p.detach().cpu().float() - adapter[ref_key].float())).item()
        return None

    def generate(self, prompt: str, max_new_tokens: int, max_input_tokens: int) -> Tuple[str, Optional[List]]:
        prompt = truncate_prompt(self.tokenizer, prompt, max_input_tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_cfg.weaver_do_sample = False
        gen_cfg.trigger_do_sample = False
        gen_cfg.temperature = 0.0
        with torch.no_grad():
            out_ids, aug_mask = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_cfg,
                return_augmentation_mask=True,
            )
        text = self.tokenizer.decode(out_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return truncate_at_first_stop(text), aug_mask.detach().cpu().tolist()

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def run_actor(actor, rows: List[Dict], args) -> List[Dict]:
    session = requests.Session()
    out_rows = []
    for idx, item in enumerate(rows):
        support = supporting_titles(item)
        prompt = build_initial_prompt(actor.tokenizer, item["question"])
        trajectory = []
        retrieved_support = set()
        final_answer = ""
        finish_reason = "max_turn"
        repeated_queries = 0
        seen_queries = set()
        first_support_hit_turn = None
        first_answer_hit_turn = None
        all_support_hit_turn = None
        no_hit_searches = 0

        for turn in range(1, args.max_turns + 1):
            step_text, aug_mask = actor.generate(prompt, args.max_new_tokens, args.max_input_tokens)
            action_type, action_text = parse_step_action(step_text)
            event = {
                "turn": turn,
                "step_text": step_text,
                "action_type": action_type,
                "action_text": action_text,
                "aug_mask": aug_mask[:1] if aug_mask is not None else None,
            }
            prompt += step_text

            if action_type == "answer":
                final_answer = action_text
                finish_reason = "answer"
                event["final_answer_hit"] = final_answer_hit(final_answer, item.get("golden_answers", []))
                trajectory.append(event)
                break

            if action_type != "search" or not action_text:
                finish_reason = "other"
                trajectory.append(event)
                break

            qn = normalize_text(action_text)
            if qn in seen_queries:
                repeated_queries += 1
                event["duplicate_query"] = True
            else:
                event["duplicate_query"] = False
                seen_queries.add(qn)

            docs = retrieve(session, action_text, args.topk)
            hits = doc_hits(docs, support, item.get("golden_answers", []))
            for t in hits["hit_support_titles"]:
                retrieved_support.add(normalize_text(t))
            if hits["hit_support_titles"] and first_support_hit_turn is None:
                first_support_hit_turn = turn
            if hits["hit_answer"] and first_answer_hit_turn is None:
                first_answer_hit_turn = turn
            if support and len(retrieved_support) >= len({normalize_text(t) for t in support}) and all_support_hit_turn is None:
                all_support_hit_turn = turn
            if not hits["hit_support_titles"] and not hits["hit_answer"]:
                no_hit_searches += 1

            event.update(hits)
            event["docs"] = [{"id": d.get("id"), "title": title_from_doc(d), "score": d.get("score")} for d in docs]
            trajectory.append(event)
            prompt += docs_to_information(docs, args.max_doc_chars)

        searches = sum(1 for e in trajectory if e["action_type"] == "search")
        summary = {
            "dataset": item["dataset"],
            "id": item.get("id"),
            "question": item["question"],
            "golden_answers": item.get("golden_answers", []),
            "supporting_titles": support,
            "finish_reason": finish_reason,
            "final_answer": final_answer,
            "final_answer_hit": final_answer_hit(final_answer, item.get("golden_answers", [])) if final_answer else False,
            "searches": searches,
            "repeated_queries": repeated_queries,
            "no_hit_searches": no_hit_searches,
            "support_coverage": (len(retrieved_support) / len({normalize_text(t) for t in support})) if support else None,
            "any_support_hit": first_support_hit_turn is not None,
            "first_support_hit_turn": first_support_hit_turn,
            "first_answer_hit_turn": first_answer_hit_turn,
            "all_support_hit_turn": all_support_hit_turn,
            "trajectory": trajectory,
        }
        out_rows.append(summary)
        print(actor.name.upper(), idx, item["dataset"], item.get("id"), "finish", finish_reason, "searches", searches, "cov", summary["support_coverage"], "final_hit", summary["final_answer_hit"])
    return out_rows


def avg(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: List[Dict]) -> Dict:
    n = len(rows)
    return {
        "n": n,
        "final_answer_hit_rate": sum(r["final_answer_hit"] for r in rows) / n if n else 0,
        "avg_searches": avg([r["searches"] for r in rows]),
        "avg_repeated_queries": avg([r["repeated_queries"] for r in rows]),
        "duplicate_query_rate_per_search": (sum(r["repeated_queries"] for r in rows) / sum(r["searches"] for r in rows)) if sum(r["searches"] for r in rows) else 0,
        "avg_no_hit_searches": avg([r["no_hit_searches"] for r in rows]),
        "no_hit_search_rate_per_search": (sum(r["no_hit_searches"] for r in rows) / sum(r["searches"] for r in rows)) if sum(r["searches"] for r in rows) else 0,
        "avg_support_coverage": avg([r["support_coverage"] for r in rows]),
        "any_support_hit_rate": sum(r["any_support_hit"] for r in rows) / n if n else 0,
        "avg_first_support_hit_turn_found_only": avg([r["first_support_hit_turn"] for r in rows]),
        "avg_first_answer_hit_turn_found_only": avg([r["first_answer_hit_turn"] for r in rows]),
        "all_support_hit_rate": sum(r["all_support_hit_turn"] is not None for r in rows) / n if n else 0,
        "avg_all_support_hit_turn_found_only": avg([r["all_support_hit_turn"] for r in rows]),
        "max_turn_failure_rate": sum(r["finish_reason"] == "max_turn" for r in rows) / n if n else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--memgen-load-model-path", default="/root/autodl-tmp/memrag_new/MemGen_checkpoints/Kana-s_MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model")
    parser.add_argument("--hotpot-path", default="/root/autodl-tmp/memrag_new/flashrag_data/hotpotqa/dev.jsonl")
    parser.add_argument("--twiki-path", default="/root/autodl-tmp/memrag_new/flashrag_data/2wikimultihopqa/dev.jsonl")
    parser.add_argument("--num-per-dataset", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    parser.add_argument("--output", default="/root/autodl-tmp/memrag_new/flashrag_memgen_multiturn_smoke.json")
    args = parser.parse_args()

    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
    print("MODEL", args.model)
    print("SEARCH_R1_URL", SEARCH_R1_URL)
    print("FLASHRAG_PIPELINE", "SearchR1Pipeline")
    print("MEMGEN_LOAD_MODEL_PATH", args.memgen_load_model_path)

    data = []
    data.extend(load_flashrag_jsonl(args.hotpot_path, args.num_per_dataset, "hotpotqa"))
    data.extend(load_flashrag_jsonl(args.twiki_path, args.num_per_dataset, "2wikimultihopqa"))

    result = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "FlashRAG SearchR1Pipeline-compatible prompt/trajectory logger",
        "datasets": ["hotpotqa/dev", "2wikimultihopqa/dev"],
        "retriever": "Search-R1 BM25 over official PeterJinGo/wiki-18 corpus via HTTP retrieve API",
        "model": args.model,
        "settings": vars(args),
        "actors": {},
    }

    base = BaseActor(args.model)
    base_rows = run_actor(base, data, args)
    base.close()
    result["actors"]["base_qwen"] = {"summary": summarize(base_rows), "rows": base_rows}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))

    memgen = MemGenActor(args.model, args.memgen_load_model_path)
    print("LORA_COMPARE_MAX_ABS_DIFF", memgen.lora_diff)
    memgen_rows = run_actor(memgen, data, args)
    lora_diff = memgen.lora_diff
    memgen.close()
    result["actors"]["trained_memgen"] = {"summary": summarize(memgen_rows), "lora_compare_max_abs_diff": lora_diff, "rows": memgen_rows}

    result["summary"] = {k: v["summary"] for k, v in result["actors"].items()}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("SUMMARY_JSON", json.dumps(result["summary"], ensure_ascii=False))
    print("RESULT_PATH", args.output)


if __name__ == "__main__":
    main()
