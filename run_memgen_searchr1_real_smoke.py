import os
import sys
import requests
import torch
from datasets import load_dataset
from transformers import GenerationConfig

MEMGEN_ROOT = "/root/autodl-tmp/memrag_new/MemGen"
sys.path.insert(0, MEMGEN_ROOT)
from memgen.model import MemGenModel  # noqa: E402

SEARCH_R1_URL = os.environ.get("SEARCH_R1_URL", "http://127.0.0.1:8000/retrieve")


def load_official_nq_question():
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split="test")
    item = ds[0]
    question = item["question"].strip()
    if not question.endswith("?"):
        question += "?"
    return question, item.get("golden_answers")


def retrieve_with_search_r1(question):
    resp = requests.post(
        SEARCH_R1_URL,
        json={"queries": [question], "topk": 3, "return_scores": True},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    hits = data["result"][0]
    if not hits:
        raise RuntimeError("Search-R1 returned no evidence")
    first_doc = hits[0]["document"]
    if "contents" not in first_doc or not first_doc["contents"]:
        raise RuntimeError("Search-R1 evidence has no contents field")
    return hits


def run_memgen_tiny(question, hits):
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    model_id = "llamafactory/tiny-random-qwen2.5"
    lora = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    config = {
        "model_name": model_id,
        "load_model_path": None,
        "attn_implementation": "eager",
        "torch_dtype": "float32",
        "max_prompt_aug_num": 1,
        "max_inference_aug_num": 1,
        "weaver": {"model_name": model_id, "prompt_latents_len": 2, "inference_latents_len": 2, "lora_config": lora},
        "trigger": {"model_name": model_id, "active": False, "lora_config": lora},
    }

    top_contents = hits[0]["document"]["contents"]
    evidence = top_contents[:900].replace("\n", " ")
    prompt = (
        "Answer using the retrieved evidence.\n"
        f"Question: {question}\n"
        f"Retrieved evidence: {evidence}"
    )

    model = MemGenModel.from_config(config).cuda().eval()
    tok = model.tokenizer
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")

    gen_cfg = GenerationConfig(
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
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

    decoded_tail = tok.decode(output_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
    print("MEMGEN_INPUT_LEN", inputs["input_ids"].shape[-1])
    print("MEMGEN_OUTPUT_SHAPE", tuple(output_ids.shape))
    print("MEMGEN_AUG_MASK", aug_mask.tolist())
    print("MEMGEN_DECODED_TAIL", decoded_tail)
    if aug_mask[0, 0].item() != 1:
        raise RuntimeError("Expected MemGen prompt latent augmentation at first generated step")


def main():
    print("HF_ENDPOINT", os.environ.get("HF_ENDPOINT"))
    question, gold = load_official_nq_question()
    print("OFFICIAL_NQ_QUESTION", question)
    print("OFFICIAL_NQ_GOLD", gold)

    hits = retrieve_with_search_r1(question)
    print("REAL_SEARCH_R1_RETRIEVE_OK")
    for rank, hit in enumerate(hits, 1):
        doc = hit["document"]
        contents = doc["contents"]
        title = contents.split("\n", 1)[0].strip('"')
        snippet = contents.split("\n", 1)[1][:220].replace("\n", " ") if "\n" in contents else contents[:220]
        print(f"RANK_{rank}_SCORE", hit["score"])
        print(f"RANK_{rank}_ID", doc.get("id"))
        print(f"RANK_{rank}_TITLE", title)
        print(f"RANK_{rank}_SNIPPET", snippet)

    run_memgen_tiny(question, hits)
    print("MEMGEN_LATENT_PATH_OK")
    print("SMOKE_TEST_PASS")


if __name__ == "__main__":
    main()
