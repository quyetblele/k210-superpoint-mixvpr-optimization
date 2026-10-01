from __future__ import annotations
import torch

from .model import K210MixVPR

def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    return score / score.mean().clamp_min(1e-12)

def _keep_top(score: torch.Tensor, count: int) -> torch.Tensor:
    if count > score.numel():
        raise ValueError("cannot keep more channels than parent owns")
    return torch.topk(score, count, largest=True).indices.sort().values

def _copy_equal_shapes(parent: K210MixVPR,
                       child: K210MixVPR) -> None:
    p = parent.state_dict()
    c = child.state_dict()
    for key, target in c.items():
        source = p.get(key)
        if source is not None and source.shape == target.shape:
            target.copy_(source)
    child.load_state_dict(c, strict=True)

def _channel_score(stage, consumer_weight: torch.Tensor) -> torch.Tensor:
    conv1_out = stage.conv1.weight.flatten(1).norm(dim=1)
    conv2_out = stage.conv2.weight.flatten(1).norm(dim=1)
    conv2_in = stage.conv2.weight.permute(1, 0, 2, 3).flatten(1).norm(dim=1)
    consumer_in = consumer_weight.movedim(1, 0).flatten(1).norm(dim=1)
    return (
        _normalize_score(conv1_out)
        + _normalize_score(conv2_out)
        + _normalize_score(conv2_in)
        + _normalize_score(consumer_in)
    )

@torch.no_grad()
def prune_w5(parent: K210MixVPR,
             child: K210MixVPR) -> dict:
    old = parent.spec.widths[4]
    new = child.spec.widths[4]
    if new >= old:
        raise ValueError("prune_w5 requires a smaller W5")
    if parent.spec.widths[3] != child.spec.widths[3]:
        raise ValueError("W4 must stay fixed while pruning W5")
    if parent.spec.mixer_hidden != child.spec.mixer_hidden:
        raise ValueError("hidden must stay fixed while pruning W5")

    _copy_equal_shapes(parent, child)
    stage_p = parent.stages[4]
    stage_c = child.stages[4]
    score = _channel_score(
        stage_p,
        parent.channel_projection.weight,
    )
    keep = _keep_top(score, new)

    stage_c.conv1.weight.copy_(stage_p.conv1.weight[keep])
    stage_c.conv1.bias.copy_(stage_p.conv1.bias[keep])
    stage_c.conv2.weight.copy_(
        stage_p.conv2.weight[keep][:, keep]
    )
    stage_c.conv2.bias.copy_(stage_p.conv2.bias[keep])
    child.channel_projection.weight.copy_(
        parent.channel_projection.weight[:, keep]
    )
    return {
        "action": "prune_w5",
        "old": int(old),
        "new": int(new),
        "kept_indices": keep.cpu().tolist(),
        "importance": "combined_l2_weight_norm",
    }

@torch.no_grad()
def prune_w3(parent: K210MixVPR,
             child: K210MixVPR) -> dict:
    old = parent.spec.widths[2]
    new = child.spec.widths[2]
    if new >= old:
        raise ValueError("prune_w3 requires a smaller W3")
    if parent.spec.widths[3:] != child.spec.widths[3:]:
        raise ValueError("W4/W5 must stay fixed while pruning W3")
    if parent.spec.mixer_hidden != child.spec.mixer_hidden:
        raise ValueError("hidden must stay fixed while pruning W3")

    _copy_equal_shapes(parent, child)
    stage_p = parent.stages[2]
    stage_c = child.stages[2]
    consumer = parent.stages[3].conv1.weight
    score = _channel_score(stage_p, consumer)
    keep = _keep_top(score, new)

    stage_c.conv1.weight.copy_(stage_p.conv1.weight[keep])
    stage_c.conv1.bias.copy_(stage_p.conv1.bias[keep])
    stage_c.conv2.weight.copy_(
        stage_p.conv2.weight[keep][:, keep]
    )
    stage_c.conv2.bias.copy_(stage_p.conv2.bias[keep])
    child.stages[3].conv1.weight.copy_(
        parent.stages[3].conv1.weight[:, keep]
    )
    return {
        "action": "prune_w3",
        "old": int(old),
        "new": int(new),
        "kept_indices": keep.cpu().tolist(),
        "importance": "combined_l2_weight_norm",
    }

@torch.no_grad()
def prune_w4(parent: K210MixVPR,
             child: K210MixVPR) -> dict:
    old = parent.spec.widths[3]
    new = child.spec.widths[3]
    if new >= old:
        raise ValueError("prune_w4 requires a smaller W4")
    if parent.spec.widths[4] != child.spec.widths[4]:
        raise ValueError("W5 must stay fixed while pruning W4")
    if parent.spec.mixer_hidden != child.spec.mixer_hidden:
        raise ValueError("hidden must stay fixed while pruning W4")

    _copy_equal_shapes(parent, child)
    stage_p = parent.stages[3]
    stage_c = child.stages[3]
    consumer = parent.stages[4].conv1.weight
    score = _channel_score(stage_p, consumer)
    keep = _keep_top(score, new)

    stage_c.conv1.weight.copy_(stage_p.conv1.weight[keep])
    stage_c.conv1.bias.copy_(stage_p.conv1.bias[keep])
    stage_c.conv2.weight.copy_(
        stage_p.conv2.weight[keep][:, keep]
    )
    stage_c.conv2.bias.copy_(stage_p.conv2.bias[keep])
    child.stages[4].conv1.weight.copy_(
        parent.stages[4].conv1.weight[:, keep]
    )
    return {
        "action": "prune_w4",
        "old": int(old),
        "new": int(new),
        "kept_indices": keep.cpu().tolist(),
        "importance": "combined_l2_weight_norm",
    }

def _hidden_score(mixer) -> torch.Tensor:
    incoming = mixer.fc1.weight.norm(dim=1)
    outgoing = mixer.fc2.weight.norm(dim=0)
    bias = mixer.fc1.bias.abs()
    return (
        _normalize_score(incoming)
        + _normalize_score(outgoing)
        + _normalize_score(bias + 1e-12)
    )

@torch.no_grad()
def prune_hidden(parent: K210MixVPR,
                 child: K210MixVPR) -> dict:
    old = parent.spec.mixer_hidden
    new = child.spec.mixer_hidden
    if new >= old:
        raise ValueError("prune_hidden requires smaller hidden width")
    if parent.spec.widths != child.spec.widths:
        raise ValueError("backbone widths must stay fixed while pruning hidden")

    _copy_equal_shapes(parent, child)
    records = []
    for index, (src, dst) in enumerate(
        zip(parent.mixers, child.mixers)
    ):
        keep = _keep_top(_hidden_score(src), new)
        dst.fc1.weight.copy_(src.fc1.weight[keep])
        dst.fc1.bias.copy_(src.fc1.bias[keep])
        dst.fc2.weight.copy_(src.fc2.weight[:, keep])
        records.append({
            "mixer": index,
            "kept_indices": keep.cpu().tolist(),
        })
    return {
        "action": "prune_hidden",
        "old": int(old),
        "new": int(new),
        "per_mixer": records,
        "importance": "combined_l2_weight_norm",
    }

def prune_one_axis(parent: K210MixVPR,
                   child: K210MixVPR,
                   action: str) -> dict:
    if parent.spec.mixer_depth != child.spec.mixer_depth:
        raise ValueError("mixer depth is invariant in V2")
    if parent.spec.descriptor_dim != 512 or child.spec.descriptor_dim != 512:
        raise ValueError("descriptor interface must remain 512D")
    if action == "prune_w5":
        return prune_w5(parent, child)
    if action == "prune_w4":
        return prune_w4(parent, child)
    if action == "prune_w3":
        return prune_w3(parent, child)
    if action == "prune_hidden":
        return prune_hidden(parent, child)
    raise ValueError(f"unsupported pruning action: {action}")
