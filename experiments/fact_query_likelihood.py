"""Fixed-Reader answer likelihood used only as a training reward."""
import math
import requests


def answer_likelihood(base_url, prompt, references):
    session=requests.Session()
    session.trust_env=False
    root=base_url.rstrip('/').removesuffix('/v1')
    def tokenize(payload):
        response=session.post(root+'/tokenize',json={'model':'Qwen3.8-27B',**payload},timeout=60)
        response.raise_for_status()
        return response.json()['tokens']
    prefix=tokenize({'messages':[{'role':'user','content':prompt}],'add_generation_prompt':True,
                     'chat_template_kwargs':{'enable_thinking':False}})
    prefix+=tokenize({'prompt':'FINAL ANSWER: ','add_special_tokens':False})
    if isinstance(references,str):
        reference=references
    else:
        reference='; '.join(references)
    target=tokenize({'prompt':reference,'add_special_tokens':False})
    response=session.post(root+'/v1/completions',json={'model':'Qwen3.8-27B','prompt':prefix+target,
        'temperature':0,'max_tokens':1,'echo':True,'logprobs':1},timeout=300)
    response.raise_for_status()
    data=response.json()
    values=data['choices'][0]['logprobs']['token_logprobs'][len(prefix):len(prefix)+len(target)]
    assert len(values)==len(target)>0 and all(v is not None and math.isfinite(v) for v in values)
    mean=sum(values)/len(values)
    return {'mean_log_probability':mean,'token_probability':math.exp(mean),
            'reference_tokens':len(target),'usage':data['usage'],'training_only':True}
