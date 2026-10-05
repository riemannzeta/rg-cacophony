"""Measurement 1: activations at the target, sense directions, cross-validated d′.

Slices S = 0..n_layers: slice L < n_layers is `blocks[L].ln1.hook_normalized` (LayerNorm output
before the affine map), slice n_layers is `ln_final.hook_normalized`. Block L maps slice L to
slice L + 1. Everything is read at the target position. Prompts are truncated right after the
target (causal model: nothing later can affect the target) and right-padded within a batch.
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformer_lens import HookedTransformer

from rgc.stimuli import Prompt

PAD_ID = 0


@dataclass
class WordIdx:
    word: str
    senses: tuple[str, str]
    trA: np.ndarray  # indices into the prompt list, ordered by pair
    trB: np.ndarray
    teA: np.ndarray
    teB: np.ndarray
    neu: np.ndarray


def word_index(prompts: list[Prompt]) -> list[WordIdx]:
    out = []
    for w in dict.fromkeys(p.word for p in prompts):
        senses = tuple(dict.fromkeys(p.sense for p in prompts if p.word == w and p.sense != "neutral"))
        pick = lambda split, sense: np.array(sorted(
            (p.pair, i) for i, p in enumerate(prompts) if p.word == w and p.split == split and p.sense == sense
        ))[:, 1]
        neu = np.array([i for i, p in enumerate(prompts) if p.word == w and p.split == "neutral"])
        out.append(WordIdx(w, senses, pick("train", senses[0]), pick("train", senses[1]),
                           pick("test", senses[0]), pick("test", senses[1]), neu))
    return out


def _batch(prompts: list[Prompt], control: bool, device: str):
    ids = [(p.control_token_ids if control else p.token_ids)[: p.target_pos + 1] for p in prompts]
    T = max(len(x) for x in ids)
    toks = torch.full((len(ids), T), PAD_ID, dtype=torch.long)
    for i, x in enumerate(ids):
        toks[i, : len(x)] = torch.tensor(x)
    tpos = torch.tensor([p.target_pos for p in prompts])
    cue = torch.zeros(len(ids), T, dtype=torch.bool)
    for i, p in enumerate(prompts):
        cue[i, p.cue_pos] = True
    return toks.to(device), tpos.to(device), cue.to(device)


def ablation_hooks(ablate: dict[int, tuple[list[int], torch.Tensor]] | None):
    """Mean-ablation of attention heads: z[:, 1:, h] <- mean_z[h] (BOS position left intact)."""
    hooks = []
    for L, (heads, means) in (ablate or {}).items():
        def hook(act, hook, heads=heads, means=means):
            act[:, 1:, heads, :] = means.to(act.dtype)
            return act
        hooks.append((f"blocks.{L}.attn.hook_z", hook))
    return hooks


@torch.no_grad()
def collect(
    model: HookedTransformer,
    prompts: list[Prompt],
    control: bool = False,
    attention: bool = True,
    slices: list[int] | None = None,
    ablate=None,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Activations at the target position.

    Returns X [N, len(slices), d]; with attention=True also H, H_nobos, M [N, L, H],
    Z [N, L, H, d_head] (head outputs before W_O) and scale [N, n_layers + 1] (LN scale per slice).
    """
    cfg, dev = model.cfg, model.cfg.device
    nL = cfg.n_layers
    slices = list(range(nL + 1)) if slices is None else slices
    names = {s: (f"blocks.{s}.ln1.hook_normalized" if s < nL else "ln_final.hook_normalized") for s in slices}
    out: dict[str, list] = {k: [] for k in ("X", "H", "H_nobos", "M", "Z", "scale")}
    for b0 in range(0, len(prompts), batch_size):
        chunk = prompts[b0 : b0 + batch_size]
        toks, tpos, cue = _batch(chunk, control, dev)
        rows = torch.arange(len(chunk), device=dev)
        store: dict[str, torch.Tensor] = {}

        def grab_target(key):
            def hook(act, hook):  # [B, T, ...] -> [B, ...] at the target position
                store[key] = act[rows, tpos].detach().clone()
            return hook

        def grab_query(key):
            def hook(act, hook):  # pattern [B, H, Tq, Tk] -> [B, H, Tk] for query = target
                store[key] = act[rows, :, tpos].detach().clone()
            return hook

        hooks = ablation_hooks(ablate)
        hooks += [(names[s], grab_target(f"X{s}")) for s in slices]
        if attention:
            for L in range(nL):
                hooks.append((f"blocks.{L}.attn.hook_pattern", grab_query(f"P{L}")))  # [B, H, Tk]
                hooks.append((f"blocks.{L}.attn.hook_z", grab_target(f"Z{L}")))  # [B, H, dh]
                hooks.append((f"blocks.{L}.ln1.hook_scale", grab_target(f"S{L}")))
            hooks.append(("ln_final.hook_scale", grab_target(f"S{nL}")))
        model.run_with_hooks(toks, return_type=None, fwd_hooks=hooks)

        out["X"].append(torch.stack([store[f"X{s}"] for s in slices], 1).double().cpu())
        if attention:
            P = torch.stack([store[f"P{L}"] for L in range(nL)], 1).double()  # [B, L, H, Tk]
            n = (tpos + 1).double()[:, None, None]
            plogp = torch.where(P > 0, P * P.log(), torch.zeros_like(P))
            out["H"].append((-plogp.sum(-1) / n.log()).cpu())
            Pn = P[..., 1:] / P[..., 1:].sum(-1, keepdim=True)
            plogp = torch.where(Pn > 0, Pn * Pn.log(), torch.zeros_like(Pn))
            out["H_nobos"].append((-plogp.sum(-1) / (n - 1).log()).cpu())
            out["M"].append((P * cue[:, None, None, :].double()).sum(-1).cpu())
            out["Z"].append(torch.stack([store[f"Z{L}"] for L in range(nL)], 1).double().cpu())
            out["scale"].append(torch.stack([store[f"S{s}"][:, 0] for s in range(nL + 1)], 1).double().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


@torch.no_grad()
def head_means(model: HookedTransformer, prompts: list[Prompt], batch_size: int = 128) -> torch.Tensor:
    """Mean of hook_z over positions 1..target_pos of the given prompts: [L, H, d_head]."""
    cfg, dev = model.cfg, model.cfg.device
    total = torch.zeros(cfg.n_layers, cfg.n_heads, cfg.d_head, dtype=torch.float64, device=dev)
    count = 0
    for b0 in range(0, len(prompts), batch_size):
        chunk = prompts[b0 : b0 + batch_size]
        toks, tpos, _ = _batch(chunk, False, dev)
        mask = torch.zeros(toks.shape, dtype=torch.bool, device=dev)
        for i, t in enumerate(tpos.tolist()):
            mask[i, 1 : t + 1] = True
        count += int(mask.sum())

        def hook(act, hook):
            L = hook.layer()
            total[L] += act[mask].double().sum(0)

        model.run_with_hooks(toks, return_type=None,
                             fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook) for L in range(cfg.n_layers)])
    return (total / count).float()


# ---------------------------------------------------------------------------------------------
# d′ analysis (numpy, float64)
# ---------------------------------------------------------------------------------------------

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1), 0.0)


def dprime(sA: np.ndarray, sB: np.ndarray) -> float:
    pooled = (sA.var(ddof=1) + sB.var(ddof=1)) / 2
    return float((sA.mean() - sB.mean()) / np.sqrt(pooled)) if pooled > 1e-24 else 0.0


def directions(X: np.ndarray, w: WordIdx) -> np.ndarray:
    """Unit sense direction per slice from the train split: [S, d]."""
    return unit(X[w.trA].mean(0) - X[w.trB].mean(0))


def dprime_curve(X: np.ndarray, w: WordIdx) -> np.ndarray:
    V = directions(X, w)
    return np.array([dprime(X[w.teA, s] @ V[s], X[w.teB, s] @ V[s]) for s in range(X.shape[1])])


def resample_counts(n: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)


def dprime_boot(X: np.ndarray, w: WordIdx, c_tr: np.ndarray, c_te: np.ndarray) -> np.ndarray:
    """Bootstrap d′ curves [B, S]; train and test PAIRS resampled with the given count matrices
    (refitting the direction each time)."""
    n_tr, n_te = c_tr.shape[1], c_te.shape[1]
    B, S = c_tr.shape[0], X.shape[1]
    out = np.zeros((B, S))
    for s in range(S):
        V = unit((c_tr @ X[w.trA, s] - c_tr @ X[w.trB, s]) / n_tr)  # [B, d]
        PA, PB = X[w.teA, s] @ V.T, X[w.teB, s] @ V.T  # [n_te, B]
        mA, mB = (c_te * PA.T).sum(1) / n_te, (c_te * PB.T).sum(1) / n_te
        vA = np.clip(((c_te * PA.T**2).sum(1) - n_te * mA**2) / (n_te - 1), 0, None)
        vB = np.clip(((c_te * PB.T**2).sum(1) - n_te * mB**2) / (n_te - 1), 0, None)
        pooled = (vA + vB) / 2
        out[:, s] = np.where(pooled > 1e-24, (mA - mB) / np.sqrt(np.where(pooled > 1e-24, pooled, 1)), 0.0)
    return out


def neutral_position(X: np.ndarray, w: WordIdx, s: int) -> np.ndarray:
    """t = (proj - m_B) / (m_A - m_B) for neutral prompts at slice s (test-split class means)."""
    v = directions(X[:, s : s + 1], w)[0]
    mA, mB = (X[w.teA, s] @ v).mean(), (X[w.teB, s] @ v).mean()
    return (X[w.neu, s] @ v - mB) / (mA - mB)


def head_contributions(Z, scale, W_O, X, widx: list[WordIdx]) -> np.ndarray:
    """c[L, h]: head's direct share of d′(L+1) on test prompts, averaged over words.

    The head output o = z W_O is mean-centred and divided by the slice-(L+1) LayerNorm scale
    (this is exactly its additive contribution to the normalised residual), projected on the
    slice-(L+1) sense direction; c = (mean_A - mean_B) / pooled SD of the full slice projection.
    """
    nL, nH = Z.shape[1], Z.shape[2]
    c = np.zeros((len(widx), nL, nH))
    for wi, w in enumerate(widx):
        V = directions(X, w)
        te = np.concatenate([w.teA, w.teB])
        for L in range(nL):
            v = V[L + 1]
            o = np.einsum("nhk,hkd->nhd", Z[te, L], W_O[L])
            o = o - o.mean(-1, keepdims=True)
            p = (o @ v) / scale[te, L + 1][:, None]
            sA, sB = X[w.teA, L + 1] @ v, X[w.teB, L + 1] @ v
            sd = np.sqrt((sA.var(ddof=1) + sB.var(ddof=1)) / 2)
            nA = len(w.teA)
            c[wi, L] = (p[:nA].mean(0) - p[nA:].mean(0)) / sd if sd > 1e-12 else 0.0
    return c.mean(0)
