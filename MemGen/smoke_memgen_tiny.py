import os
import torch
from transformers import GenerationConfig
from memgen.model import MemGenModel


def main():
    print('HF_ENDPOINT', os.environ.get('HF_ENDPOINT'))
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)
    if torch.cuda.is_available():
        print('gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

    model_id = 'llamafactory/tiny-random-qwen2.5'
    lora = {
        'r': 2,
        'lora_alpha': 4,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'lora_dropout': 0.0,
        'bias': 'none',
        'task_type': 'CAUSAL_LM',
    }
    config = {
        'model_name': model_id,
        'load_model_path': None,
        'attn_implementation': 'eager',
        'torch_dtype': 'float32',
        'max_prompt_aug_num': 1,
        'max_inference_aug_num': 1,
        'weaver': {'model_name': model_id, 'prompt_latents_len': 2, 'inference_latents_len': 2, 'lora_config': lora},
        'trigger': {'model_name': model_id, 'active': False, 'lora_config': lora},
    }

    model = MemGenModel.from_config(config).cuda().eval()
    tok = model.tokenizer
    messages = [{'role': 'user', 'content': 'What is 1+1?'}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt')
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
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            generation_config=gen_cfg,
            return_augmentation_mask=True,
        )

    print('input_len', inputs['input_ids'].shape[-1])
    print('output_shape', tuple(output_ids.shape))
    print('aug_mask', aug_mask.tolist())
    print('decoded_tail', tok.decode(output_ids[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=False))
    if aug_mask[0, 0].item() != 1:
        raise RuntimeError('Expected prompt latent augmentation at first generated step')
    print('SMOKE_TEST_PASS')


if __name__ == '__main__':
    main()
