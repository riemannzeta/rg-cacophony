"""Constants added by PREREG-2.md (kept separate so the PREDICTIONS.md config hash is unchanged)."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config2:
    # A. anaesthesia analogue
    betas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)
    windows: tuple[int | None, ...] = (None, 64, 16, 4)  # None = intact
    local_response_floor: float = 0.5
    # B. conflicting / graded cues
    n_frames: int = 16
    n_train_frames: int = 8
    n_cues_per_sense: int = 8
    n_neutral_fillers: int = 8
    bc_threshold: float = 0.555
    # C. Pythia vs Mamba
    mamba_pairs: tuple[tuple[str, str], ...] = (("state-spaces/mamba-130m-hf", "pythia-160m"),
                                                ("state-spaces/mamba-370m-hf", "pythia-410m"))
    mamba_depth_stride: int = 2
    mamba_shuffle_seeds: tuple[int, ...] = (1001, 1002, 1003)
    # D. universality
    scale_models: tuple[str, ...] = ("pythia-160m", "pythia-410m", "pythia-1b")


CFG2 = Config2()


def config2_hash() -> str:
    return hashlib.sha256(json.dumps(asdict(CFG2), sort_keys=True, default=str).encode()).hexdigest()[:16]
