import numpy as np

from rgc import chimera as ch


def test_chimera_high_when_some_communities_synchronise_and_others_do_not():
    rng = np.random.default_rng(0)
    S, T, d = 5, 1 + 64, 32
    X = rng.normal(size=(S, T, d))
    sync = X.copy()
    sync[:, 1:33] = rng.normal(size=(S, 1, d)) + 0.05 * rng.normal(size=(S, 32, d))  # half the tokens aligned
    com = ch.blocks(T, 16)  # 4 communities: 2 synchronised, 2 random
    assert ch.stats(sync, com)["chimera"] > 10 * ch.stats(X, com)["chimera"]


def test_eff_dim_low_for_low_rank_cloud():
    rng = np.random.default_rng(1)
    S, T, d = 2, 101, 64
    low = np.repeat(rng.normal(size=(S, T, 2)) @ rng.normal(size=(2, d)), 1, axis=0)
    full = rng.normal(size=(S, T, d))
    assert ch.stats(low, ch.blocks(T, 10))["eff_dim"][1] < 3 < ch.stats(full, ch.blocks(T, 10))["eff_dim"][1]


def test_sentence_partition():
    toks = ["<bos>", " The", " cat", " sat", ".", " A", " dog", " ran", " far", "!", " x"]
    com = ch.sentences(toks)
    assert [list(c) for c in com] == [[1, 2, 3, 4], [5, 6, 7, 8, 9]]
