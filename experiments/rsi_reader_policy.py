"""Learn evidence selection over QFAD candidates with a fixed answering model."""
from dataclasses import dataclass
from pathlib import Path
import torch


FEATURES = ('rerank', 'query_similarity', 'graph_rank', 'quoted_evidence',
            'selected_source', 'requested_date', 'current_state_recency', 'compactness')


@dataclass
class Selection:
    indices: list[int]
    log_probability: torch.Tensor
    document_tokens: int


class ReaderPolicy:
    def __init__(self):
        self.weights = torch.nn.Parameter(torch.tensor([5., 2., .5, .2, .5, 1., .3, .1]))
        self.optimizer = torch.optim.Adam([self.weights], lr=.02)
        self.updates = 0
        self.paths = {}

    def remember(self, episode, documents, reward):
        previous = self.paths.get(episode)
        if previous is None or reward > previous['reward']:
            self.paths[episode] = {'documents': list(documents), 'reward': float(reward)}

    def replay(self, episode, documents, costs, budget):
        path = self.paths.get(episode)
        if path is None or any(doc not in documents for doc in path['documents']):
            return None
        indices = [documents.index(doc) for doc in path['documents']]
        cost = sum(costs[i] for i in indices)
        if cost > budget:
            return None
        return Selection(indices, self.weights.sum() * 0, cost)

    def select(self, features, costs, budget, limit=15, seed=None):
        matrix = torch.as_tensor(features, dtype=torch.float32)
        assert matrix.ndim == 2 and matrix.shape[1] == len(FEATURES)
        assert len(costs) == len(matrix) and all(c > 0 for c in costs)
        logits = matrix @ self.weights
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        remaining = list(range(len(matrix)))
        chosen = []
        consumed = 0
        log_probability = logits.sum() * 0
        for _ in range(limit):
            eligible = [i for i in remaining if consumed + costs[i] <= budget]
            if not eligible:
                break
            log_probs = torch.log_softmax(logits[eligible], dim=0)
            if generator is None:
                position = int(torch.argmax(log_probs))
            else:
                position = int(torch.multinomial(log_probs.exp(), 1, generator=generator))
            index = eligible[position]
            log_probability = log_probability + log_probs[position]
            chosen.append(index)
            consumed += costs[index]
            remaining.remove(index)
        return Selection(chosen, log_probability, consumed)

    def update(self, selections, rewards, baseline_reward):
        assert len(selections) == len(rewards) and selections
        advantages = [float(r) - float(baseline_reward) for r in rewards]
        before = self.weights.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([-a * s.log_probability / max(1, len(s.indices))
                            for s, a in zip(selections, advantages)]).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([self.weights], 1.)
        assert torch.isfinite(norm)
        if any(a != 0 for a in advantages):
            self.optimizer.step()
            self.updates += 1
        return {'baseline_reward': float(baseline_reward), 'rewards': list(map(float, rewards)),
                'advantages': advantages, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                'weights_before': before.tolist(), 'weights_after': self.weights.detach().tolist(),
                'parameter_delta_l2': float(torch.linalg.vector_norm(self.weights.detach() - before)),
                'updates': self.updates}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'features': FEATURES, 'weights': self.weights.detach(),
                    'optimizer': self.optimizer.state_dict(), 'updates': self.updates,
                    'paths': self.paths}, path)

    def load(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert tuple(saved['features']) == FEATURES
        with torch.no_grad():
            self.weights.copy_(saved['weights'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.updates = saved['updates']
        self.paths = saved.get('paths', {})
