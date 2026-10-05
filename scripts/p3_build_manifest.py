"""PREREG-3: build the fresh-passage manifest (WikiText-103-raw TRAIN split, first 64 articles with
>= 256 body tokens, dataset order; same parser/detokeniser/truncation as the original manifest)."""

import json

from transformers import AutoTokenizer

from rgc import wikitext
from rgc.config import CFG
from rgc.determinism import REPO

RULE = ("PREREG-3 fresh passages: WikiText-103-raw TRAIN split, articles in dataset order (first 200,000 "
        "lines parsed), first 64 with >= 256 body tokens; same header parser, detokeniser and truncation "
        "(first 256 tokens) as stimuli/wikitext_manifest.json. None used by any earlier analysis.")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(CFG.hf_repo, revision=f"step{CFG.final_step}")
    m = wikitext.build_manifest(tok, splits=("train",), max_lines=200_000, rule=RULE)
    old = json.loads(wikitext.MANIFEST_PATH.read_text())
    old_ids = {json.dumps(p["token_ids"]) for p in old["passages"]}
    assert not any(json.dumps(p["token_ids"]) in old_ids for p in m["passages"]), "overlap with original passages"
    path = REPO / "stimuli" / "wikitext_fresh_manifest.json"
    path.write_text(json.dumps(m, indent=1))
    print(len(m["passages"]), "fresh passages ->", path, "| sha256_all", m["sha256_all"][:16])
    print("first titles:", [p["title"] for p in m["passages"][:5]])


if __name__ == "__main__":
    main()
