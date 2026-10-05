"""Seeds, deterministic kernels, and an environment fingerprint for provenance."""

import os

# Must be set before CUDA initialises for deterministic cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import importlib.metadata as md
import platform
import random
import socket
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def disable_native_overrides() -> list[str]:
    """Route every op to the standard ATen kernels.

    torch >= 2.14 can override some eager CUDA ops (bmm outer products, norms, sums, ...) with
    Triton/CuTe-DSL kernels compiled at run time. We disable all of them: the measurements should
    run on ATen's documented (and, with use_deterministic_algorithms, deterministic) kernels.
    Returns the DSL names that were disabled (recorded in env_info).
    """
    try:
        from torch._native import registry
    except ImportError:
        return []
    dsls = sorted({n.dsl_name for nodes in registry._graphs.values() for n in nodes})
    if dsls:
        registry.deregister_op_overrides(disable_dsl_names=dsls)
    return dsls


_DISABLED_NATIVE: list[str] = []


def set_determinism(seed: int = 0) -> None:
    global _DISABLED_NATIVE
    _DISABLED_NATIVE = disable_native_overrides() or _DISABLED_NATIVE
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def env_info() -> dict:
    from rgc.config import config_hash

    pkgs = ["torch", "transformer-lens", "transformers", "numpy", "scipy", "pandas", "datasets"]
    return {
        "git_sha": git_sha(),
        "config_hash": config_hash(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
        "packages": {p: md.version(p) for p in pkgs},
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "native_overrides_disabled": _DISABLED_NATIVE,
    }
