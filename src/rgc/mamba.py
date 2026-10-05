"""PREREG-2 part C: Mamba (HF MambaForCausalLM, pure-torch path) behind the M2 perturbation interface.

Residual slice L (0..n_layers) is the hidden state entering `backbone.layers[L]`; slice n_layers is
the input to `backbone.norm_f`. Perturbation and reading use forward pre-hooks. Clean baselines are
row-matched, as for Pythia (same batch shape, same kernels). No fused kernels are used: neither the
`kernels` nor the `mamba_ssm` package is installed, so HF falls back to its torch implementation.
"""

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rgc.models import shuffle_parameters_


def _patch_selective_scan() -> None:
    """Make HF's torch selective scan keep its input dtype.

    `mamba_selective_scan` casts B and the conv output with `.float()`. For float32 inputs (the
    measured models) that is a no-op; for the float64 reference it silently quantises the perturbed
    signal to float32 and corrupts the eps = 1e-6 finite-difference reference (D18). The patch replaces
    the two casts by `.to(input_dtype)`; tests verify the float32 path is bitwise unchanged.
    """
    import inspect

    from transformers.models.mamba import modeling_mamba as mm

    if getattr(mm, "_rgc_patched", False):
        return
    src = inspect.getsource(mm.mamba_selective_scan)
    assert src.count(".float()") == 2, "unexpected HF Mamba source; re-check the patch"
    src = src.replace(".float()", ".to(input_dtype)")
    ns: dict = {}
    exec(compile(src, mm.__file__, "exec"), mm.__dict__, ns)
    mm._rgc_original_selective_scan = mm.mamba_selective_scan
    mm.mamba_selective_scan = ns["mamba_selective_scan"]
    mm._rgc_patched = True


_patch_selective_scan()


class MambaRunner:
    def __init__(self, repo: str | None = None, device: str = "cpu", shuffle_seed: int | None = None,
                 dtype=torch.float32, model=None, tokenizer=None):
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype)
            tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = model.to(device).eval()
        if shuffle_seed is not None:
            shuffle_parameters_(self.model, shuffle_seed)
        self.tokenizer = tokenizer
        c = self.model.config
        self.cfg = SimpleNamespace(n_layers=c.num_hidden_layers, d_model=c.hidden_size, device=device, dtype=dtype)
        bb = self.model.backbone
        self._points = list(bb.layers) + [bb.norm_f]  # pre-hook targets for slices 0..n

    def to_float64(self) -> "MambaRunner":
        """Float64 reference copy (D18). HF Mamba silently computes parts of a float64 model in float32:
        blocks cast the residual to float32 (residual_in_fp32), MambaRMSNorm normalises in float32, and the
        selective scan casts B and its input with .float() (patched module-wide above). Each would put a
        float32 noise floor on the eps = 1e-6 finite-difference reference. All three are switched to keep
        float64 here; the measured float32 model is unchanged (each cast is a no-op for float32)."""
        import types

        from transformers.models.mamba.modeling_mamba import MambaRMSNorm

        def rmsnorm_keep_dtype(mod, hidden_states):  # HF's MambaRMSNorm always computes in float32
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            return mod.weight * (hidden_states * torch.rsqrt(variance + mod.variance_epsilon))

        self.model = self.model.to(torch.float64)
        for block in self.model.backbone.layers:
            block.residual_in_fp32 = False
        for mod in self.model.modules():
            if isinstance(mod, MambaRMSNorm):
                mod.forward = types.MethodType(rmsnorm_keep_dtype, mod)
        self.cfg.dtype = torch.float64
        return self

    def _rows(self, tokens, R):
        if tokens.dim() == 1:
            return tokens.to(self.cfg.device)[None].expand(R, -1).contiguous()
        return tokens.to(self.cfg.device)

    @torch.no_grad()
    def _run(self, x, pre_hooks):
        handles = [self._points[i].register_forward_pre_hook(h) for i, h in pre_hooks]
        try:
            self.model(input_ids=x, use_cache=False)
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def clean_stack(self, tokens, k: int, R: int) -> torch.Tensor:
        store = {}

        def grab(i):
            def hook(mod, args):
                store[i] = args[0].detach().clone()
            return hook

        self._run(self._rows(tokens, R), [(i, grab(i)) for i in range(k, self.cfg.n_layers + 1)])
        return torch.stack([store[i] for i in range(k, self.cfg.n_layers + 1)], 1)

    @torch.no_grad()
    def perturb_response(self, tokens, k, p, u, eps, clean=None, extra_hooks=None) -> torch.Tensor:
        assert not extra_hooks, "attention manipulations do not apply to Mamba"
        dev, R = self.cfg.device, p.shape[0]
        p, u, eps = p.to(dev), u.to(dev), eps.to(dev, torch.float32)
        if clean is None:
            clean = self.clean_stack(tokens, k, R)
        n_sl = self.cfg.n_layers + 1 - k
        out = torch.empty(R, n_sl, tokens.shape[-1], device=dev, dtype=torch.float32)
        rows = torch.arange(R, device=dev)

        def inject(mod, args):
            h = args[0].clone()
            site = h[rows, p]
            h[rows, p] = site + (eps * site.norm(dim=-1))[:, None].to(h.dtype) * u.to(h.dtype)
            return (h,) + tuple(args[1:])

        def read(i):
            def hook(mod, args):
                ref = clean[:, i - k]
                out[:, i - k] = ((args[0] - ref).norm(dim=-1) / ref.norm(dim=-1)).float()
            return hook

        hooks = [(k, inject)] + [(i, read(i)) for i in range(k, self.cfg.n_layers + 1)]
        self._run(self._rows(tokens, R), hooks)
        return out

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=tokens.to(self.cfg.device), use_cache=False).logits
