"""Frozen analysis constants. Every number that shapes a comparison lives here.

Changing anything in this file after PREDICTIONS.md is committed requires an entry in
DECISIONS.md. `config_hash()` is written into every result file so a summary can be traced to
the exact constants that produced it.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class Config:
    # --- models -------------------------------------------------------------------------------
    model_name: str = "pythia-160m"
    hf_repo: str = "EleutherAI/pythia-160m"
    checkpoints: tuple[int, ...] = (0, 512, 4000, 143000)
    final_step: int = 143000
    shuffle_seeds: tuple[int, ...] = (1001, 1002, 1003)
    dtype: str = "float32"
    prepend_bos: bool = True  # BOS = <|endoftext|> (id 0) at sequence position 0

    # --- Measurement 1 ------------------------------------------------------------------------
    candidate_words: tuple[str, ...] = (
        "bank", "bat", "crane", "pitcher", "seal", "spring", "match", "mole", "pen", "bark",
    )
    n_per_sense: int = 30
    n_train_per_sense: int = 20  # remaining 10 are test; split is by cue vocabulary (disjoint)
    n_neutral: int = 12
    n_continuations: int = 4
    control_token: str = " thing"
    k_heads: int = 6
    n_null_sets: int = 200
    null_seed: int = 7

    # --- Measurement 2 ------------------------------------------------------------------------
    n_passages: int = 64
    passage_tokens: int = 256  # content tokens; sequence length is passage_tokens + 1 with BOS
    wikitext_config: str = "wikitext-103-raw-v1"
    wikitext_splits: tuple[str, ...] = ("test", "validation")  # in this order (D9)
    # Injection positions are SEQUENCE positions (BOS is position 0).
    p_grid: tuple[int, ...] = (16, 64, 128)
    n_dirs: int = 8
    dir_seed: int = 11
    # ε grid: 1e-6 … 1 in half-decade steps (13 values).
    eps_grid: tuple[float, ...] = tuple(float(10.0**e) for e in np.arange(-6.0, 0.01, 0.5))
    n_calib_passages: int = 8
    n_calib_dirs: int = 4
    linear_tol: float = 0.05

    theta_primary: float = 0.1
    theta_grid: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    degenerate_p1_lo: float = 0.01
    degenerate_p1_hi: float = 0.99
    degenerate_frac: float = 0.90
    min_downstream_layers: int = 3  # primary PCI uses k <= n_layers - 3
    saturation_frac: float = 0.9
    z_primary_k: int = 0
    z_primary_p: int = 64

    gating_before: int = -2  # k = L* - 2
    gating_after: int = 1  # k = L* + 1
    gating_window: int = 3

    # --- statistics ---------------------------------------------------------------------------
    n_boot: int = 10_000
    boot_seed: int = 12345
    master_seed: int = 0

    extra: dict = field(default_factory=dict)


CFG = Config()


def config_hash(cfg: Config = CFG) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
