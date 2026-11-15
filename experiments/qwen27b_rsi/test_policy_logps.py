"""Check the policy's causal action-token loss without an optimizer update."""
from types import SimpleNamespace

import torch

from policy_worker import logps


class TokenLogits(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.arange(40, device='cuda', dtype=torch.float32).reshape(5, 8) / 10)

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
        assert input_ids.tolist() == [[1, 2, 3, 5, 6]]
        assert logits_to_keep == 3
        return SimpleNamespace(logits=self.values[-logits_to_keep:].unsqueeze(0))


def main():
    model = TokenLogits()
    before = model.values.detach().clone()
    actual = logps(model, [1, 2, 3], [5, 6])
    expected = torch.log_softmax(model.values.detach()[2:4], -1)[torch.arange(2), torch.tensor([5, 6])]
    torch.testing.assert_close(actual, expected)
    (-actual.mean()).backward()
    assert torch.count_nonzero(model.values.grad[:2]) == 0
    assert torch.count_nonzero(model.values.grad[4:]) == 0
    assert (model.values.grad[2:4].abs().sum(-1) > 0).all()
    torch.testing.assert_close(model.values, before, rtol=0, atol=0)
    print('PASS: causal alignment, action-only loss positions, unchanged parameters; no optimizer update')


if __name__ == '__main__':
    main()
