"""Batched frozen-writer inference for identical source inputs across adapters."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from fact_writer_protocol import messages
from fact_writer_worker import attach, state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--adapters',nargs='+',required=True,help='name=adapter_path')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    requests=json.loads(args.inputs.read_text())
    torch.set_num_threads(4);torch.manual_seed(17)
    torch.cuda.set_per_process_memory_fraction(.20)
    weights='/home/administrator/du/model/Qwen/Qwen3-4B-Instruct-2507'
    tokenizer=AutoTokenizer.from_pretrained(weights,local_files_only=True,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(weights,local_files_only=True,dtype=torch.bfloat16,
                                             attn_implementation='sdpa',device_map={'':'cuda:0'})
    attach(model);model.eval()
    parameter_map=dict(model.named_parameters())
    expected=set(state(model))
    config=GenerationConfig(do_sample=False,max_new_tokens=3072,use_cache=True,
                            eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    for condition in args.adapters:
        name,path=condition.split('=',1)
        adapter=torch.load(path,map_location='cpu',weights_only=True)
        assert set(adapter)==expected
        with torch.no_grad():
            for key,value in adapter.items():parameter_map[key].copy_(value)
        directory=args.out/name;directory.mkdir(parents=True,exist_ok=True)
        pending=[r for r in requests if not (directory/f'{r["episode_id"]}.json').exists()]
        for offset in range(0,len(pending),args.batch_size):
            batch=pending[offset:offset+args.batch_size]
            prompts=[tokenizer.apply_chat_template(messages(r['sources']),tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False) for r in batch]
            assert all(len(p)<=6000 for p in prompts)
            encoded=tokenizer.pad({'input_ids':prompts},padding=True,return_tensors='pt').to('cuda')
            width=encoded['input_ids'].shape[1]
            with torch.inference_mode():
                outputs=model.generate(**encoded,generation_config=config)
            for request,prompt,tensor in zip(batch,prompts,outputs):
                tokens=tensor[width:].tolist();finished=tokenizer.eos_token_id in tokens
                if finished:tokens=tokens[:tokens.index(tokenizer.eos_token_id)+1]
                result={'episode_id':request['episode_id'],'adapter_path':path,'condition':name,
                    'text':tokenizer.decode(tokens,skip_special_tokens=True),'finished':finished,
                    'prompt_tokens':len(prompt),'completion_tokens':len(tokens),'token_ids':tokens}
                (directory/f'{request["episode_id"]}.json').write_text(json.dumps(result,ensure_ascii=False))
            print(name,'completed',offset+len(batch),'of',len(pending),flush=True)


if __name__=='__main__':main()
