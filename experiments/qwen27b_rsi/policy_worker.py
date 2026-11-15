"""Qwen27B role policy with clipped action updates and a frozen base reference."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import sys
import time

import torch


def emit(packet):
    print(json.dumps(packet, ensure_ascii=False), flush=True)


@contextmanager
def use_policy_snapshot(model, snapshot):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    current = {name: p.detach().clone() for name, p in parameters.items()}
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(snapshot[name])
        yield
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(current[name])


def logps(model, prompt, tokens):
    storage = torch.autograd.graph.save_on_cpu(pin_memory=True) if torch.is_grad_enabled() else nullcontext()
    with storage:
        ids = torch.tensor([prompt + tokens], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                           logits_to_keep=len(tokens) + 1).logits[:, :-1].float()
        targets = torch.tensor(tokens, device='cuda')
        return -torch.nn.functional.cross_entropy(logits[0], targets, reduction='none')


def action_logits(model, prompt, tokens):
    ids = torch.tensor([prompt + tokens], device=next(model.parameters()).device)
    with torch.autocast(ids.device.type, dtype=torch.bfloat16):
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                     logits_to_keep=len(tokens) + 1).logits[0, :-1]


def guided_logps(model, prompt, tokens, teacher_prompt):
    """Full-vocabulary reverse KL at sampled output positions, with a detached teacher."""
    with torch.no_grad():
        teacher = action_logits(model, teacher_prompt, tokens).cpu()
    with torch.autograd.graph.save_on_cpu(pin_memory=next(model.parameters()).is_cuda):
        logits = action_logits(model, prompt, tokens)
        targets = torch.tensor(tokens, device=logits.device)
        chosen, kl = [], []
        for offset in range(0, len(tokens), 32):
            student_log = logits[offset:offset + 32].float().log_softmax(-1)
            teacher_log = teacher[offset:offset + 32].to(student_log).log_softmax(-1)
            chosen.append(student_log.gather(1, targets[offset:offset + 32, None]).flatten())
            kl.append((student_log.exp() * (student_log - teacher_log)).sum(-1))
        return torch.cat(chosen), torch.cat(kl)


def encode_guided_turns(paths, guidance, groups, encode):
    """Bind feedback to the exact sampled action and its original branch observation."""
    turns = {(a['group'], a['row']) for path in paths for a in path}
    teachers = {}
    for item in guidance:
        key = item['group'], item['row']
        if key not in turns or key in teachers or not item['hint'].strip():
            raise ValueError('Unknown, duplicate or empty guidance target')
        messages = item['messages']
        if messages[-1]['role'] != 'user' or encode(messages) != groups[key[0]][0]:
            raise ValueError('Hint targets a different write observation')
        conditioned = [*messages[:-1], {**messages[-1], 'content':
            messages[-1]['content'] + '\nTraining feedback for this write:\n' + item['hint']}]
        teachers[key] = encode(conditioned)
    return teachers


def main():
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--update-epochs', type=int, default=1)
    args = parser.parse_args()
    assert args.update_epochs > 0
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    config = AutoConfig.from_pretrained(args.model).text_config
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', key_mapping={r'^model\.language_model\.': 'model.'},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys'), loading
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=8,
        lora_alpha=16, target_modules='all-linear', lora_dropout=0., bias='none'))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert all('lora_' in n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.)
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['parameters'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
    groups = {}
    reader_snapshot, reader_step = None, None
    emit({'event': 'ready', 'pid': os.getpid(), 'model': args.model,
          'trainable_parameters': sum(p.numel() for p in parameters), 'step': step,
          'update_epochs': args.update_epochs, 'saved_tensor_storage': 'cpu'})
    for line in sys.stdin:
        request = json.loads(line)
        operation = request['operation']
        if operation == 'close':
            emit({'event': 'closed', 'step': step})
            break
        if operation == 'discard_samples':
            for key in request['groups']:
                del groups[key]
            emit({'event': 'discarded'})
            continue
        if operation == 'restore_samples':
            packet = request['samples']
            assert packet['policy_step'] == step, 'Saved actions must match the resumed policy'
            prompt = tokenizer.apply_chat_template(request['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            assert len(prompt) == packet['prompt_tokens']
            assert packet['id'] == request['id'] and packet['rows']
            groups[request['id']] = (prompt, packet['rows'], step)
            emit({'event': 'samples_restored', 'id': request['id'], 'policy_step': step,
                  'count': len(packet['rows'])})
            continue
        if operation == 'freeze_reader':
            reader = torch.load(request['checkpoint'], map_location='cpu', weights_only=False)
            reader_snapshot = {name: reader['parameters'][name].to(device=p.device, dtype=p.dtype)
                               for name, p in model.named_parameters() if p.requires_grad}
            reader_step = reader['step']
            del reader
            emit({'event': 'reader_frozen', 'checkpoint': request['checkpoint'], 'step': reader_step})
            continue
        if operation in ('sample', 'infer', 'infer_reader'):
            started = time.monotonic()
            prompt = tokenizer.apply_chat_template(request['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            max_tokens = request['max_tokens']
            if operation in ('infer', 'infer_reader'):
                remaining = request['native_budget'] - len(prompt)
                if len(prompt) > 12000 or remaining <= 0:
                    emit({'event': 'budget_exhausted', 'id': request['id'],
                          'prompt_tokens': len(prompt),
                          'policy_step': reader_step if operation == 'infer_reader' else step})
                    continue
                max_tokens = min(max_tokens, remaining)
            if len(prompt) > 12000:
                raise ValueError('Policy observation exceeds the configured 12000-token budget')
            model.eval()
            torch.manual_seed(request['seed'])
            ids = torch.tensor([prompt], device='cuda')
            generation = ({'do_sample': True, 'temperature': 1., 'top_p': 1., 'top_k': 0,
                           'num_return_sequences': request['count']} if operation == 'sample'
                          else {'do_sample': False, 'num_return_sequences': 1})
            if operation == 'infer_reader' and reader_snapshot is None:
                raise ValueError('Reader snapshot has not been loaded')
            role = use_policy_snapshot(model, reader_snapshot) if operation == 'infer_reader' else nullcontext()
            with role, torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    **generation, max_new_tokens=max_tokens,
                    use_cache=True, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id)
            rows = []
            for output in outputs:
                tokens = output[len(prompt):].tolist()
                finished = tokenizer.eos_token_id in tokens
                if finished:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                rows.append({'text': tokenizer.decode(tokens, skip_special_tokens=True),
                             'token_ids': tokens, 'finished': finished})
            if operation == 'sample':
                groups[request['id']] = (prompt, rows, step)
            packet = {'event': 'samples', 'id': request['id'], 'rows': rows,
                      'prompt_tokens': len(prompt), 'elapsed_seconds': time.monotonic() - started,
                      'policy_step': reader_step if operation == 'infer_reader' else step,
                      'seed': request['seed'], 'role': 'frozen_reader' if operation == 'infer_reader' else 'current_policy'}
            (args.out / f'{request["id"]}.samples.json').write_text(json.dumps(packet))
            emit(packet)
        elif operation == 'update':
            torch.cuda.empty_cache()
            advantages = request['advantages']
            paths = request.get('trajectories', [[{'group': request['id'], 'row': i}]
                                                for i in range(len(advantages))])
            assert len(paths) == len(advantages)
            guidance = request.get('turn_guidance')
            if 'hints' in request:  # Historical single-write clients.
                if guidance is not None or any(len(path) != 1 for path in paths):
                    raise ValueError('Use turn_guidance for a multi-write update')
                guidance = [{**path[0], 'messages': request['messages'], 'hint': hint}
                            for path, hint in zip(paths, request['hints'], strict=True) if hint]
            teacher_prompts = {}
            if guidance is not None:
                if args.update_epochs != 1 or not request['guidance_weight'] > 0:
                    raise ValueError('Targeted guidance requires one on-policy epoch and positive weight')
                def encode(msgs):
                    return tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                         enable_thinking=False, return_dict=False)
                teacher_prompts = encode_guided_turns(paths, guidance, groups, encode)
            trajectories = []
            consumed = set()
            for path in paths:
                actions = []
                for action in path:
                    prompt, rows, sampled_step = groups[action['group']]
                    assert sampled_step == step, 'Policy changed inside an on-policy trajectory'
                    actions.append((prompt, rows[action['row']], action.get('credit_weight', 1.)))
                    consumed.add(action['group'])
                trajectories.append(actions)
            before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            model.eval()
            with torch.no_grad():
                old = [[logps(model, prompt, row['token_ids']).detach()
                        for prompt, row, _ in actions] for actions in trajectories]
                with model.disable_adapter():
                    reference = [[logps(model, prompt, row['token_ids']).detach()
                                  for prompt, row, _ in actions] for actions in trajectories]
            model.train()
            epoch_metrics = []
            for epoch in range(args.update_epochs):
                optimizer.zero_grad(set_to_none=True)
                losses, divergences, ratio_changes = [], [], []
                guidance_loss = 0.
                for path, actions, advantage, previous_steps, reference_steps in zip(paths, trajectories, advantages, old, reference):
                    token_count = sum(len(row['token_ids']) for _, row, _ in actions)
                    for action, (prompt, row, weight), previous, ref in zip(path, actions, previous_steps, reference_steps):
                        assert weight > 0, 'Step credit must preserve the trajectory advantage sign'
                        teacher_prompt = teacher_prompts.get((action['group'], action['row']))
                        if teacher_prompt is not None:
                            current, divergence = guided_logps(model, prompt, row['token_ids'], teacher_prompt)
                        else:
                            current = logps(model, prompt, row['token_ids'])
                        ratio = (current - previous).exp()
                        surrogate = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                        log_ratio = ref - current
                        kl = log_ratio.exp() - log_ratio - 1.
                        loss = (-weight * surrogate + .02 * kl).sum() / (len(trajectories) * token_count)
                        if teacher_prompt is not None:
                            term = divergence.sum() / (len(trajectories) * token_count)
                            guidance_loss += term.detach().item()
                            loss = loss + request['guidance_weight'] * term
                        loss.backward()
                        losses.append(loss.item())
                        divergences.append(kl.detach().mean().item())
                        ratio_changes.append((ratio.detach() - 1).abs().mean().item())
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                assert torch.isfinite(norm), norm
                optimizer.step()
                epoch_metrics.append({'epoch': epoch + 1, 'loss': sum(losses),
                                      'mean_reference_kl': sum(divergences) / len(divergences),
                                      'mean_ratio_change_from_sample_policy': sum(ratio_changes) / len(ratio_changes),
                                      'gradient_norm': norm.item()})
            step += 1
            after = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            delta = sum((after[n] - before[n]).square().sum().item() for n in after) ** .5
            temp_checkpoint = args.out / 'checkpoint.pt.tmp'
            torch.save({'parameters': after, 'optimizer': optimizer.state_dict(), 'step': step}, temp_checkpoint)
            temp_checkpoint.replace(args.out / 'checkpoint.pt')
            model.save_pretrained(args.out / 'adapter')
            for group in consumed:
                del groups[group]
            packet = {'event': 'updated', 'id': request['id'], 'step': step,
                      'advantages': advantages, 'rewards': request['rewards'],
                      'loss': sum(losses), 'mean_reference_kl': sum(divergences) / len(divergences),
                      'gradient_norm': norm.item(), 'adapter_delta_l2': delta,
                      'optimizer_steps_in_update': args.update_epochs, 'epochs': epoch_metrics,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
            if guidance is not None:
                packet.update(guided_turns=len(teacher_prompts), targeted_reverse_kl=guidance_loss,
                              guidance_weight=request['guidance_weight'])
            (args.out / f'{request["id"]}.update.json').write_text(json.dumps(packet))
            emit(packet)
        else:
            raise ValueError(operation)


if __name__ == '__main__':
    main()
