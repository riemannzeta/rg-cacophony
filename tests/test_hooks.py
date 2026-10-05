import torch


def test_bos_and_shapes(model, tokens):
    assert tokens[0].item() == model.tokenizer.bos_token_id
    T, cfg = tokens.shape[0], model.cfg
    _, cache = model.run_with_cache(tokens[None].to(cfg.device))
    for L in range(cfg.n_layers):
        assert cache[f"blocks.{L}.hook_resid_pre"].shape == (1, T, cfg.d_model)
        assert cache[f"blocks.{L}.ln1.hook_normalized"].shape == (1, T, cfg.d_model)
        assert cache[f"blocks.{L}.attn.hook_pattern"].shape == (1, cfg.n_heads, T, T)
        assert cache[f"blocks.{L}.attn.hook_z"].shape == (1, T, cfg.n_heads, cfg.d_head)
    assert cache["ln_final.hook_normalized"].shape == (1, T, cfg.d_model)
    pat = cache["blocks.3.attn.hook_pattern"][0]
    torch.testing.assert_close(pat.sum(-1), torch.ones_like(pat.sum(-1)))
    assert torch.all(pat.triu(1) == 0)  # causal


def test_ln_normalized_is_unit_rms(model, tokens):
    _, cache = model.run_with_cache(tokens[None].to(model.cfg.device))
    x = cache["blocks.5.ln1.hook_normalized"][0]
    torch.testing.assert_close(x.mean(-1), torch.zeros(x.shape[0], device=x.device), atol=1e-5, rtol=0)
    torch.testing.assert_close(x.pow(2).mean(-1).sqrt(), torch.ones(x.shape[0], device=x.device), atol=1e-3, rtol=0)


def test_tl_matches_hf(model, tokens):
    from rgc.models import Variant, load_hf

    hf = load_hf(Variant(143000)).to(model.cfg.device)
    with torch.no_grad():
        lt = model(tokens[None].to(model.cfg.device))
        lh = hf(tokens[None].to(model.cfg.device)).logits
    rel = ((lt - lh).abs().max() / lh.abs().max()).item()
    assert rel < 1e-5, rel
