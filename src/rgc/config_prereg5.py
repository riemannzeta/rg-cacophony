"""Constants added by PREREG-5.md (confirmatory test of the lambda zero-crossing across scale/family)."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config5:
    manifest: str = "stimuli/wikitext_fresh3_manifest.json"
    out_root: str = "results/m2_fresh3"
    n_passages: int = 32
    n_dirs: int = 4
    calib_passages: int = 4
    calib_dirs: int = 2
    # model -> (run name stem, k_stride); k <= n_layers - 3 in every case
    models: tuple[tuple[str, int], ...] = (("pythia-160m", 1), ("pythia-410m", 2), ("pythia-1b", 1), ("gpt2", 1))
    betas_trained: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    betas_shuffled: tuple[float, ...] = (0.25, 1.0, 2.0)
    shuffle_seed: int = 1001
    crossing_low: float = 0.75
    crossing_high: float = 1.5


CFG5 = Config5()


def run_name(model: str, beta: float = 1.0, shuffled: bool = False, step0: bool = False) -> str:
    base = model if model == "gpt2" else f"{model}_step{0 if step0 else 143000}"
    if shuffled:
        base += f"-shuffled{CFG5.shuffle_seed}"
    return base if beta == 1.0 else f"{base}__beta={beta:g}"


def config5_hash() -> str:
    return hashlib.sha256(json.dumps(asdict(CFG5), sort_keys=True, default=str).encode()).hexdigest()[:16]
