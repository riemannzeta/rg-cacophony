"""Constants added by PREREG-3.md (fresh-data replication of the anaesthesia pattern)."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config3:
    manifest: str = "stimuli/wikitext_fresh_manifest.json"
    out_root: str = "results/m2_fresh"
    theta_primary: float = 0.01
    dm_frac: float = 0.2
    betas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)
    windows_predicted: tuple[int, ...] = (64, 16)  # R4: intact > 64 > 16; w = 4 has no prediction
    windows_run: tuple[int, ...] = (64, 16, 4)


CFG3 = Config3()


def config3_hash() -> str:
    return hashlib.sha256(json.dumps(asdict(CFG3), sort_keys=True, default=str).encode()).hexdigest()[:16]
