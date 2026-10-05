"""Measurement 1 causal check: mean-ablate selected heads, re-measure Δd′, compare with nulls."""

import numpy as np
import torch
from transformer_lens import HookedTransformer

from rgc.m1_order import WordIdx, collect, dprime_curve
from rgc.stimuli import Prompt


def select_heads(score: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Top-k (layer, head) by score [L, H]; ties broken by (layer, head) order."""
    order = np.argsort(-score.reshape(-1), kind="stable")[:k]
    return [divmod(int(i), score.shape[1]) for i in order]


def spec(heads: list[tuple[int, int]], means: torch.Tensor) -> dict:
    by_layer: dict[int, list[int]] = {}
    for L, h in heads:
        by_layer.setdefault(L, []).append(h)
    return {L: (hs, means[L, hs]) for L, hs in by_layer.items()}


def final_delta(model: HookedTransformer, prompts: list[Prompt], widx: list[WordIdx],
                heads, means, batch_size: int = 128) -> float:
    """Pooled Δd′ at the final slice with the given heads mean-ablated; directions refit."""
    nL = model.cfg.n_layers
    ab = spec(heads, means) if heads else None
    Xt = collect(model, prompts, control=False, attention=False, slices=[nL], ablate=ab, batch_size=batch_size)["X"]
    Xc = collect(model, prompts, control=True, attention=False, slices=[nL], ablate=ab, batch_size=batch_size)["X"]
    return float(np.mean([dprime_curve(Xt, w)[0] - dprime_curve(Xc, w)[0] for w in widx]))


def random_sets(rng: np.random.Generator, n: int, k: int, nL: int, nH: int) -> list[list[tuple[int, int]]]:
    return [[divmod(int(i), nH) for i in rng.choice(nL * nH, size=k, replace=False)] for _ in range(n)]


def layer_matched_sets(rng: np.random.Generator, n: int, selected, nH: int) -> list[list[tuple[int, int]]]:
    """Same multiset of layers as `selected`; random heads within those layers, excluding selected."""
    sel = set(selected)
    layers: dict[int, int] = {}
    for L, _ in selected:
        layers[L] = layers.get(L, 0) + 1
    out = []
    for _ in range(n):
        s = []
        for L, cnt in sorted(layers.items()):
            pool = [h for h in range(nH) if (L, h) not in sel]
            s += [(L, int(h)) for h in rng.choice(pool, size=min(cnt, len(pool)), replace=False)]
        out.append(s)
    return out
