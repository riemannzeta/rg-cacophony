"""Constants added by PREREG-4.md (criticality and chimera states on a second fresh passage set)."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config4:
    manifest: str = "stimuli/wikitext_fresh2_manifest.json"
    out_root: str = "results/m2_fresh2"
    theta: float = 0.01
    dm_frac: float = 0.2
    betas: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
    near_natural: tuple[float, ...] = (0.75, 1.0, 1.5)
    beta_sweep_models: tuple[str, ...] = ("", "-shuffled1001")  # suffix on pythia-160m_step143000
    community_block: int = 16
    waist_middle: tuple[int, int] = (4, 8)  # inclusive slice range (C3)


CFG4 = Config4()


def config4_hash() -> str:
    return hashlib.sha256(json.dumps(asdict(CFG4), sort_keys=True, default=str).encode()).hexdigest()[:16]
