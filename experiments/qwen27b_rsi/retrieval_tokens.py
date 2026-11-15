"""Count retrieval input tokens with the tokenizers used by the running service."""
from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer


TOKENIZERS = Path(__file__).resolve().parents[3] / 'results/qwen27b_rsi/tokenizers'


@lru_cache(maxsize=2)
def tokenizer(model):
    return AutoTokenizer.from_pretrained(TOKENIZERS / model, local_files_only=True)


def count_tokens(texts, model, add_special_tokens=True):
    total = 0
    for start in range(0, len(texts), 128):
        rows = tokenizer(model)(texts[start:start + 128], padding=False,
                                add_special_tokens=add_special_tokens)['input_ids']
        total += sum(len(row) for row in rows)
    return total


def rerank_tokens(payload):
    return count_tokens([payload['query'] + doc for doc in payload['documents']],
                        'Qwen3-Reranker-4B', add_special_tokens=False)
