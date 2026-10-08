"""PREREG-4 part C: chimera / metastability indices and effective dimensionality across depth.

Following Shanahan (2010), with depth (residual slices 1..n) playing the role of time and token
positions the role of oscillators. Per slice L, the LayerNorm-normalised residuals of positions
1..T-1 (BOS excluded) are mean-centred across positions (removing the shared anisotropic direction)
and unit-normalised. Communities are fixed partitions of positions (primary: contiguous blocks of
16 tokens; secondary: sentences).

  R_c(L)        = || mean_{q in c} u_q(L) ||                    synchrony of community c
  chimera       = mean_L  Var_c R_c(L)                          (Shanahan's chi)
  metastability = mean_c  Var_L R_c(L)                          (Shanahan's lambda)
  eff_dim(L)    = participation ratio of the centred, unit-normalised token cloud
  global_sync(L)= || mean_q x_q(L)/||x_q(L)|| ||  (uncentred; anisotropy, reported only)
"""

import numpy as np
import torch


def blocks(n_tokens: int, size: int) -> list[np.ndarray]:
    """Contiguous communities over positions 1..n_tokens-1 (BOS excluded)."""
    pos = np.arange(1, n_tokens)
    return [pos[i : i + size] for i in range(0, len(pos) - size + 1, size)]


def sentences(token_strs: list[str], min_len: int = 4) -> list[np.ndarray]:
    """Sentence communities: split after tokens ending in . ! ? (positions 1.., BOS excluded)."""
    out, cur = [], []
    for q in range(1, len(token_strs)):
        cur.append(q)
        if token_strs[q].strip().endswith((".", "!", "?")):
            if len(cur) >= min_len:
                out.append(np.array(cur))
            cur = []
    if len(cur) >= min_len:
        out.append(np.array(cur))
    return out


def stats(X: np.ndarray, communities: list[np.ndarray]) -> dict:
    """X: [S, T, d] normalised residuals for one passage (slice 0 = embeddings)."""
    S = X.shape[0]
    R = np.zeros((S, len(communities)))
    eff, gsync = np.zeros(S), np.zeros(S)
    for L in range(S):
        x = X[L, 1:].astype(np.float64)
        gsync[L] = np.linalg.norm((x / np.linalg.norm(x, axis=1, keepdims=True)).mean(0))
        c = x - x.mean(0, keepdims=True)
        u = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        s = np.linalg.svd(u - u.mean(0, keepdims=True), compute_uv=False) ** 2
        eff[L] = s.sum() ** 2 / (s**2).sum() if s.sum() > 0 else 0.0
        full = np.zeros((X.shape[1], X.shape[2]))
        full[1:] = u
        for j, com in enumerate(communities):
            R[L, j] = np.linalg.norm(full[com].mean(0))
    Rt = R[1:]  # depth "time" excludes the context-free embedding slice
    return {"chimera": float(Rt.var(axis=1).mean()), "metastability": float(Rt.var(axis=0).mean()),
            "eff_dim": eff.tolist(), "global_sync": gsync.tolist(), "R_mean": Rt.mean(1).tolist()}


@torch.no_grad()
def normalised_residuals(model, tokens: torch.Tensor, hooks=()) -> np.ndarray:
    """[S, T, d] LayerNorm-normalised residuals at every slice for one sequence (TransformerLens)."""
    nL = model.cfg.n_layers
    names = [f"blocks.{L}.ln1.hook_normalized" for L in range(nL)] + ["ln_final.hook_normalized"]
    store = {}

    def grab(act, hook):
        store[hook.name] = act[0].detach().float().cpu().numpy()

    model.run_with_hooks(tokens[None].to(model.cfg.device), return_type=None,
                         fwd_hooks=list(hooks) + [(n, grab) for n in names])
    return np.stack([store[n] for n in names])
