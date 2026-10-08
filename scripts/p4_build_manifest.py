"""PREREG-4: second fresh-passage manifest. WikiText-103-raw TRAIN split, articles in dataset order
(first 200,000 lines parsed), qualifying articles 65-128 (i.e. the 64 that follow the PREREG-3 set);
same parser, detokeniser and truncation as before. Asserts no overlap with any earlier passage set."""

import json

from transformers import AutoTokenizer

from rgc import wikitext
from rgc.config import CFG
from rgc.determinism import REPO

RULE = ("PREREG-4 fresh passages: WikiText-103-raw TRAIN split, dataset order (first 200,000 lines), "
        "qualifying articles (>= 256 body tokens) number 65-128, i.e. the 64 following the PREREG-3 set; "
        "same header parser, detokeniser and truncation. None used by any earlier analysis.")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(CFG.hf_repo, revision=f"step{CFG.final_step}")
    import rgc.config as cfgmod
    from dataclasses import replace
    big = replace(CFG, n_passages=128)
    cfgmod_CFG, wikitext.CFG = wikitext.CFG, big  # collect 128, keep the second 64
    try:
        m = wikitext.build_manifest(tok, splits=("train",), max_lines=200_000, rule=RULE)
    finally:
        wikitext.CFG = cfgmod_CFG
    import hashlib
    second = m["passages"][64:]
    for i, p in enumerate(second):
        p["index"] = i
    ids = [p["token_ids"] for p in second]
    m.update(passages=second, sha256_all=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    seen = set()
    for f in ("wikitext_manifest.json", "wikitext_fresh_manifest.json"):
        seen |= {json.dumps(p["token_ids"]) for p in json.loads((REPO / "stimuli" / f).read_text())["passages"]}
    assert not any(json.dumps(x) in seen for x in ids), "overlap with earlier passages"
    fresh3 = json.loads((REPO / "stimuli" / "wikitext_fresh_manifest.json").read_text())["passages"]
    assert [p["header_line"] for p in m["passages"]][0] > fresh3[-1]["header_line"]
    path = REPO / "stimuli" / "wikitext_fresh2_manifest.json"
    path.write_text(json.dumps(m, indent=1))
    print(len(second), "passages ->", path.name, "| sha256_all", m["sha256_all"][:16], "| first:", second[0]["title"])


if __name__ == "__main__":
    main()
