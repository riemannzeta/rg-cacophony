"""Stimulus integrity: saved files match the source, positions are right, splits are disjoint."""

from collections import defaultdict

import pytest
from transformers import AutoTokenizer

from rgc import stimuli, wikitext
from rgc.config import CFG


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(CFG.hf_repo, revision=f"step{CFG.final_step}")


@pytest.fixture(scope="module")
def prompts():
    return stimuli.load()


def test_saved_file_matches_source(tok, prompts):
    built, report = stimuli.build(tok)
    assert report["errors"] == []
    assert [p.__dict__ for p in built] == [p.__dict__ for p in prompts]


def test_positions_and_tokens(tok, prompts):
    ctrl = tok(CFG.control_token, add_special_tokens=False)["input_ids"][0]
    for p in prompts:
        assert p.token_ids == [tok.bos_token_id] + tok(p.text, add_special_tokens=False)["input_ids"]
        assert p.token_ids[p.target_pos] == tok(" " + p.word, add_special_tokens=False)["input_ids"][0]
        assert p.control_token_ids[p.target_pos] == ctrl
        diff = [i for i, (a, b) in enumerate(zip(p.token_ids, p.control_token_ids)) if a != b]
        assert diff == [p.target_pos]
        assert all(0 < c < p.target_pos for c in p.cue_pos)
        assert (len(p.cue_pos) > 0) == (p.split != "neutral")
        assert p.n_tokens - p.target_pos - 1 >= 28  # continuation for the gating test


def test_minimal_pairs_share_frame_and_continuation(prompts):
    groups = defaultdict(list)
    for p in prompts:
        if p.split != "neutral":
            groups[(p.word, p.split, p.pair)].append(p)
    for g in groups.values():
        assert len(g) == 2 and g[0].sense != g[1].sense
        assert g[0].frame == g[1].frame and g[0].continuation == g[1].continuation


def test_counts(prompts):
    for w in {p.word for p in prompts}:
        ps = [p for p in prompts if p.word == w]
        senses = {p.sense for p in ps} - {"neutral"}
        assert len(senses) == 2
        for s in senses:
            assert sum(p.sense == s and p.split == "train" for p in ps) == CFG.n_train_per_sense
            assert sum(p.sense == s and p.split == "test" for p in ps) == CFG.n_per_sense - CFG.n_train_per_sense
        assert sum(p.split == "neutral" for p in ps) == CFG.n_neutral


def test_wikitext_manifest_integrity():
    ids = wikitext.load_passages()
    assert len(ids) == CFG.n_passages and all(len(x) == CFG.passage_tokens for x in ids)
