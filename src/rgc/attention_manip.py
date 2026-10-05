"""Attention manipulations for the PREREG-2 "anaesthesia" tests (applied at hook_attn_scores).

TransformerLens applies the causal mask (-inf) before hook_attn_scores, so masked entries stay
masked under both manipulations.

- flatten(beta): scores <- beta * scores on unmasked entries (inverse temperature). beta = 1 is the
  intact model; beta = 0 gives uniform causal attention: everything still reaches everything, but
  every query averages its context identically (integration without differentiation).
- local_window(w): each query q attends only to keys k with q - w < k <= q, plus BOS (k = 0) so the
  attention sink survives. Caps cross-position reach by construction (a positive control).
"""

import torch


def flatten_hooks(beta: float, layers):
    def hook(scores, hook):
        finite = torch.isfinite(scores)
        return torch.where(finite, scores * beta, scores)
    return [(f"blocks.{L}.attn.hook_attn_scores", hook) for L in layers]


def window_hooks(w: int, layers):
    def hook(scores, hook):
        T = scores.shape[-1]
        q = torch.arange(T, device=scores.device)[:, None]
        k = torch.arange(T, device=scores.device)[None, :]
        far = (k <= q - w) & (k > 0)
        return scores.masked_fill(far, float("-inf"))
    return [(f"blocks.{L}.attn.hook_attn_scores", hook) for L in layers]


def manipulation_hooks(spec: str | None, n_layers: int):
    """'none' | 'beta=<x>' | 'window=<w>' -> list of hooks applied to every layer."""
    if spec in (None, "none"):
        return []
    kind, val = spec.split("=")
    layers = range(n_layers)
    if kind == "beta":
        return flatten_hooks(float(val), layers)
    if kind == "window":
        return window_hooks(int(val), layers)
    raise ValueError(spec)
