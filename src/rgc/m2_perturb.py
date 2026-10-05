"""Perturbation engine for Measurement 2.

A perturbation is injected into the residual stream at (layer k, sequence position p):
    resid_pre[k][p] += eps * ||resid_pre[k][p]|| * u        (u: unit, mean-removed)
and the response is read at every later residual slice L in {k, ..., n_layers} and every position:
    r(L, q) = ||resid'[L][q] - resid[L][q]|| / ||resid[L][q]||.

Residual slice index L means `blocks.L.hook_resid_pre` for L < n_layers and
`blocks.{n-1}.hook_resid_post` for L = n_layers (the input to ln_final).

Clean activations are computed with the *same batch shape* and compared row by row, so every row
of the perturbed batch is compared against a clean row that went through identical kernels. This
makes eps = 0 give a bitwise-zero response, and q < p give exactly zero (causality).
"""

import torch
from transformer_lens import HookedTransformer


def unit_directions(d_model: int, n: int, seed: int) -> torch.Tensor:
    """n random unit vectors in R^d with zero mean across coordinates (LN discards the mean)."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, d_model, generator=g, dtype=torch.float64)
    u = u - u.mean(dim=1, keepdim=True)
    u = u / u.norm(dim=1, keepdim=True)
    return u.to(torch.float32)


def center_unit(v: torch.Tensor) -> torch.Tensor:
    v = v.to(torch.float64)
    v = v - v.mean()
    return (v / v.norm()).to(torch.float32)


def slice_names(n_layers: int, k: int) -> list[str]:
    names = [f"blocks.{L}.hook_resid_pre" for L in range(k, n_layers)]
    return names + [f"blocks.{n_layers - 1}.hook_resid_post"]


def _rows(tokens: torch.Tensor, batch: int, device) -> torch.Tensor:
    """[T] -> `batch` identical rows; [R, T] is used as is (one sequence per row)."""
    if tokens.dim() == 1:
        return tokens.to(device)[None].expand(batch, -1).contiguous()
    assert tokens.shape[0] == batch
    return tokens.to(device)


@torch.no_grad()
def clean_stack(model: HookedTransformer, tokens: torch.Tensor, k: int, batch: int,
                extra_hooks=None) -> torch.Tensor:
    """Clean residual slices L = k..n for the batch rows: [batch, n+1-k, T, d].

    extra_hooks: manipulation hooks (e.g. attention flattening) applied to clean and perturbed
    runs alike, so the response is always relative to the manipulated model's own clean state.
    """
    names = slice_names(model.cfg.n_layers, k)
    store: dict[str, torch.Tensor] = {}

    def grab(act, hook):
        store[hook.name] = act.detach().clone()

    x = _rows(tokens, batch, model.cfg.device)
    model.run_with_hooks(x, return_type=None, fwd_hooks=list(extra_hooks or []) + [(n, grab) for n in names])
    return torch.stack([store[n] for n in names], dim=1)


@torch.no_grad()
def perturb_response(
    model: HookedTransformer,
    tokens: torch.Tensor,  # [T] (replicated per row) or [R, T]
    k: int,
    p: torch.Tensor,
    u: torch.Tensor,
    eps: torch.Tensor,
    clean: torch.Tensor | None = None,
    extra_hooks=None,
) -> torch.Tensor:
    """Relative response lattices r[row, L-k, q] for a batch of perturbations at layer k.

    p:   [R] long, injection position per row
    u:   [R, d] unit directions
    eps: [R] relative magnitudes
    clean: optional precomputed clean_stack(model, tokens, k, R)
    """
    dev = model.cfg.device
    R = p.shape[0]
    p, u, eps = p.to(dev), u.to(dev), eps.to(dev, torch.float32)
    if clean is None:
        clean = clean_stack(model, tokens, k, R, extra_hooks)
    assert clean.shape[0] == R
    names = slice_names(model.cfg.n_layers, k)
    out = torch.empty(R, len(names), tokens.shape[-1], device=dev, dtype=torch.float32)
    rows = torch.arange(R, device=dev)

    def inject(act, hook):
        site = act[rows, p]  # [R, d]
        act[rows, p] = site + (eps * site.norm(dim=-1))[:, None] * u
        return act

    def read(i):
        def hook(act, hook):
            ref = clean[:, i]
            out[:, i] = ((act - ref).norm(dim=-1) / ref.norm(dim=-1)).float()

        return hook

    hooks = list(extra_hooks or []) + [(f"blocks.{k}.hook_resid_pre", inject)]
    hooks += [(n, read(i)) for i, n in enumerate(names)]
    model.run_with_hooks(_rows(tokens, R, dev), return_type=None, fwd_hooks=hooks)
    return out
