"""Model variants: Pythia checkpoints and weight-shuffled controls, loaded unprocessed in float32.

Unprocessed (no LN folding, no centering of writing weights) so that the residual stream and its
norms are the original model's, not a basis-shifted equivalent.
"""

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from rgc.config import CFG


@dataclass(frozen=True)
class Variant:
    step: int
    shuffle_seed: int | None = None
    model_name: str = CFG.model_name

    @property
    def label(self) -> str:
        base = self.model_name if not is_pythia(self.model_name) else f"{self.model_name}@step{self.step}"
        return base if self.shuffle_seed is None else f"{base}-shuffled{self.shuffle_seed}"


def registry(model_name: str = CFG.model_name) -> list[Variant]:
    out = [Variant(s, None, model_name) for s in CFG.checkpoints]
    out += [Variant(CFG.final_step, seed, model_name) for seed in CFG.shuffle_seeds]
    return out


def is_pythia(model_name: str) -> bool:
    return model_name.startswith("pythia")


def hf_repo(model_name: str) -> str:
    if is_pythia(model_name):
        return f"EleutherAI/{model_name}"
    if model_name == "gpt2":  # PREREG-5: second model family (no training checkpoints)
        return "openai-community/gpt2"
    raise ValueError(model_name)


@torch.no_grad()
def shuffle_parameters_(module: torch.nn.Module, seed: int) -> None:
    """Permute every parameter tensor's entries independently, in place.

    Preserves each tensor's multiset of values (norms, moments, histograms) while destroying the
    learned arrangement. Buffers (e.g. rotary inv_freq) are not parameters and are left alone.
    """
    g = torch.Generator().manual_seed(seed)
    for _, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        flat = p.detach().reshape(-1)
        perm = torch.randperm(flat.numel(), generator=g)
        p.copy_(flat[perm].reshape(p.shape))


def load_hf(variant: Variant) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo(variant.model_name), dtype=torch.float32,
        **({"revision": f"step{variant.step}"} if is_pythia(variant.model_name) else {}),
    )
    model.eval()
    if variant.shuffle_seed is not None:
        shuffle_parameters_(model, variant.shuffle_seed)
    return model


def load(variant: Variant, device: str = "cpu") -> HookedTransformer:
    # Weights always come from `hf` (the HF model at revision step{N}, shuffled if requested).
    # Do NOT pass checkpoint_value: TransformerLens 3.9 then re-downloads the checkpoint and
    # silently ignores hf_model, which would turn every shuffled control back into the original.
    hf = load_hf(variant)
    model = HookedTransformer.from_pretrained_no_processing(
        variant.model_name,
        hf_model=hf,
        dtype=torch.float32,
        device=device,
        default_prepend_bos=False,  # we prepend BOS ourselves (see tokenize)
    )
    model.eval()
    del hf
    return model


def tokenize(model: HookedTransformer, text: str) -> torch.Tensor:
    """BOS + text tokens, as a 1-D LongTensor. BOS handling is explicit, never implicit."""
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    if CFG.prepend_bos:
        ids = [model.tokenizer.bos_token_id] + ids
    return torch.tensor(ids, dtype=torch.long)
