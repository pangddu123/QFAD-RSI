"""Exercise Qwen27B NF4 loading and one masked LoRA optimizer step."""
import argparse
import json
from pathlib import Path
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    started = time.monotonic()
    config = AutoConfig.from_pretrained(args.model).text_config
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa',
        key_mapping={r'^model\.language_model\.': 'model.'},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        output_loading_info=True,
    )
    assert not loading['missing_keys'], loading['missing_keys']
    assert not loading.get('mismatched_keys'), loading.get('mismatched_keys')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type='CAUSAL_LM', r=8, lora_alpha=16,
        target_modules='all-linear', lora_dropout=0.0, bias='none'))
    model.config.use_cache = False
    model.train()
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert parameters and all('lora_' in name for name, _ in parameters)
    frozen = next(p for p in model.parameters() if not p.requires_grad)
    frozen_before = frozen.detach().reshape(-1)[:64].clone()
    probe = next(p for name, p in parameters if 'lora_B' in name)
    before = probe.detach().clone()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [
        {'role': 'system', 'content': 'You are the Writer. Return one JSON graph operation supported by the source.'},
        {'role': 'user', 'content': 'Source s1: Patient: I stopped taking empagliflozin. Graph: no medication events.'},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    action = json.dumps({'operation': 'ADD_FACT', 'fact': {
        'subject': 'patient', 'relation': 'stopped taking', 'object': 'empagliflozin',
        'status': 'stopped', 'citations': [{'source_id': 's1',
                                         'quote': 'Patient: I stopped taking empagliflozin.'}]}})
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    action_ids = tokenizer.encode(action, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = torch.tensor([prompt_ids + action_ids], device='cuda')
    labels = ids.clone()
    labels[:, :len(prompt_ids)] = -100
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=1e-5)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
    assert torch.isfinite(loss), loss
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], 1.0)
    assert torch.isfinite(gradient_norm) and gradient_norm > 0, gradient_norm
    optimizer.step()
    delta = (probe.detach() - before).abs().max().item()
    assert delta > 0, 'Adapter did not change'
    assert torch.equal(frozen_before, frozen.detach().reshape(-1)[:64])
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    torch.cuda.synchronize()
    result = {
        'model': args.model, 'seed': 17, 'device': torch.cuda.get_device_name(),
        'loss': loss.item(), 'gradient_norm': gradient_norm.item(),
        'adapter_max_delta': delta, 'frozen_base_probe_unchanged': True,
        'trainable_parameters': sum(p.numel() for _, p in parameters),
        'tokens': ids.shape[1], 'action_tokens': len(action_ids),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'elapsed_seconds': time.monotonic() - started,
        'purpose': 'Training-kernel witness; this is not an RL result or a task-trained checkpoint',
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print('WITNESS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
