from __future__ import annotations
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners

_TASK = losses.MultiSimilarityLoss(alpha=1.0, beta=50.0, base=0.0)
_MINER = miners.MultiSimilarityMiner(epsilon=0.1)

def retrieval_loss(descriptors: torch.Tensor, labels: torch.Tensor):
    pairs = _MINER(descriptors, labels)
    return _TASK(descriptors, labels, pairs)


def retrieval_loss_with_stats(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
):
    pairs = _MINER(descriptors, labels)
    loss = _TASK(descriptors, labels, pairs)
    stats = {
        "positive_pairs": int(pairs[0].numel()),
        "negative_pairs": int(pairs[2].numel()),
    }
    return loss, stats

def ranking_kd(student: torch.Tensor, teacher: torch.Tensor,
               labels: torch.Tensor, temperature: float = 0.07):
    s = student @ student.T
    t = teacher @ teacher.T
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~diagonal
    negative = ~same
    log_s = F.log_softmax(s.masked_fill(diagonal, -1e4) / temperature, dim=1)
    prob_t = F.softmax(t.masked_fill(diagonal, -1e4) / temperature, dim=1)
    pos_score = t.masked_fill(~positive, -1e4).max(1).values
    neg_score = t.masked_fill(~negative, -1e4).max(1).values
    confidence = (pos_score - neg_score).clamp(0.0, 1.0).detach()
    row_kl = F.kl_div(log_s, prob_t, reduction="none").sum(1)
    return (row_kl * (0.25 + confidence)).mean()
