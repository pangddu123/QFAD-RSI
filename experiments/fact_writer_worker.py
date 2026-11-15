"""Token-level LoRA REINFORCE with query rewards supplied by a local controller."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from fact_writer_protocol import messages, WRITER_PROMPT


class LoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(8, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, 8, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=5**.5)

    def forward(self, x):
        return self.base(x) + 2 * ((x.float() @ self.a.T) @ self.b.T).to(x.dtype)


def attach(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            parent, _, field = name.rpartition('.')
            setattr(model.get_submodule(parent), field, LoRA(module))
    return [p for p in model.parameters() if p.requires_grad]


def state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def completion_logps(model, prompt, output):
    ids = torch.tensor([prompt + output], device='cuda')
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=len(output)+1).logits[:, :-1].float()
    target = torch.tensor(output, device='cuda')
    return -nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target, reduction='none')


def generate(model, tokenizer, prompt, count, cap, greedy=False):
    model.eval()
    config = GenerationConfig(do_sample=not greedy, temperature=1., top_p=1., top_k=0,
                              max_new_tokens=cap, num_return_sequences=count, use_cache=True,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    ids = torch.tensor([prompt], device='cuda')
    with torch.no_grad():
        outputs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)
    rows = []
    for output in outputs:
        tokens = output[len(prompt):].tolist()
        finished = tokenizer.eos_token_id in tokens
        if finished:
            tokens = tokens[:tokens.index(tokenizer.eos_token_id)+1]
        rows.append({'text': tokenizer.decode(tokens, skip_special_tokens=True), 'token_ids': tokens, 'finished': finished})
    return rows


def emit(data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--greedy-every', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(args.inputs.read_text())
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.cuda.set_per_process_memory_fraction(.38)
    weights = '/home/administrator/du/model/Qwen/Qwen3-4B-Instruct-2507'
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(weights, local_files_only=True, dtype=torch.bfloat16,
                                                attn_implementation='sdpa', device_map={'': 'cuda:0'})
    trainable = attach(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=0.)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.out / 'checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['adapter'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step'] + 1
        torch.set_rng_state(checkpoint['rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
    else:
        torch.save(state(model), args.out / 'initial_adapter.pt')
    emit({'event': 'ready', 'pid': os.getpid(), 'trainable_parameters': sum(p.numel() for p in trainable),
          'model_path':weights,'writer_prompt':WRITER_PROMPT,'sampling_temperature':1.,'samples_per_group':4,
          'max_new_tokens':3072,'learning_rate':5e-5,'greedy_every':args.greedy_every,'start_step':start,
          'base_parameters_frozen': all(not p.requires_grad for n,p in model.named_parameters() if not n.endswith(('.a','.b')))})
    for step in range(start, args.steps):
        item = inputs[step % len(inputs)]
        prompt = tokenizer.apply_chat_template(messages(item['sources']), tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False, return_dict=False)
        if len(prompt) > 6000:
            raise ValueError('Writer source prompt exceeds 6000 tokens')
        evaluate_greedy=step==0 or (step+1)%args.greedy_every==0 or step==args.steps-1
        rollout_path = args.out / f'rollouts_{step}.json'
        if args.resume and rollout_path.exists():
            packet = json.loads(rollout_path.read_text())
            assert packet['episode_id'] == item['episode_id'] and packet['prompt_tokens'] == len(prompt)
            rows = packet['rows']
        else:
            # Independent group seeds also make continuation independent of cached sampling.
            torch.manual_seed(17 + step)
            greedy = generate(model, tokenizer, prompt, 1, 3072, greedy=True)[0] if evaluate_greedy else None
            rows = generate(model, tokenizer, prompt, 4, 3072)
            packet = {'event': 'rollouts', 'step': step, 'episode_id': item['episode_id'],
                      'initial_greedy': greedy, 'rows': rows, 'prompt_tokens': len(prompt), 'sampling_seed':17+step}
            rollout_path.write_text(json.dumps(packet, ensure_ascii=False))
        emit(packet)
        response = json.loads(sys.stdin.readline())
        assert response['step'] == step and response['episode_id'] == item['episode_id']
        rewards = response['rewards']
        assert len(rewards) == len(rows)
        advantages = response['advantages']
        assert len(advantages) == len(rows)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        before = state(model)
        losses = []
        for row, advantage in zip(rows, advantages):
            loss = -advantage * completion_logps(model, prompt, row['token_ids']).sum() / (len(rows)*3072)
            loss.backward()
            losses.append(loss.item())
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite writer gradient')
        optimizer.step()
        after = state(model)
        delta = sum(float((after[n]-before[n]).square().sum()) for n in after)**.5
        torch.save({'adapter': after, 'optimizer': optimizer.state_dict(), 'step': step,
                    'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, args.out / 'checkpoint.pt')
        torch.save(after,args.out/f'adapter_step{step+1}.pt')
        post = generate(model, tokenizer, prompt, 1, 3072, greedy=True)[0] if evaluate_greedy else None
        result = {'event': 'update', 'step': step, 'episode_id': item['episode_id'], 'rewards': rewards,
                  'advantages': advantages, 'adapter_delta_l2': delta, 'gradient_norm': norm.item(),
                  'loss': sum(losses), 'post_update_greedy': post,
                  'query_reward_components':response.get('query_reward_components'),
                  'normalized_query_components':response.get('normalized_query_components'),
                  'query_signal_span':response.get('query_signal_span'),
                  'query_signal_active':response.get('query_signal_active'),
                  'peak_memory_bytes': torch.cuda.max_memory_allocated()}
        (args.out / f'update_{step}.json').write_text(json.dumps(result, ensure_ascii=False))
        emit(result)
    emit({'event': 'complete', 'steps': args.steps})


if __name__ == '__main__':
    main()
