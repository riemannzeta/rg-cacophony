"""PREREG-5: third fresh-passage manifest. WikiText-103-raw TRAIN split, dataset order (first 200,000
lines), qualifying articles after number 128 (i.e. after the PREREG-3 and PREREG-4 sets) that have
>= 256 body tokens under BOTH the Pythia (GPT-NeoX) and the GPT-2 tokenizers; the first 64 such.
Stores Pythia token ids (first 256) and the detokenised body text (for re-tokenising with GPT-2).
Asserts no overlap with any earlier passage set."""

import hashlib
import json
from dataclasses import replace

from transformers import AutoTokenizer

from rgc import wikitext
from rgc.config import CFG
from rgc.determinism import REPO

RULE = ("PREREG-5 fresh passages: WikiText-103-raw TRAIN split, dataset order (first 200,000 lines); "
        "qualifying articles numbered > 128 (after the PREREG-3 and PREREG-4 sets) with >= 256 body tokens under "
        "both the Pythia and GPT-2 tokenizers; first 64. Same parser/detokeniser; body text stored for GPT-2.")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(CFG.hf_repo, revision=f"step{CFG.final_step}")
    g2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", CFG.wikitext_config, split="train")
    lines = ds[:200_000]["text"]
    qualifying, out = 0, []
    for art in wikitext.articles(lines):
        body = wikitext.detokenize("".join(art["lines"])).strip()
        ids = tok(body, add_special_tokens=False)["input_ids"]
        if len(ids) < CFG.passage_tokens:
            continue
        qualifying += 1
        if qualifying <= 128:
            continue
        if len(g2(body, add_special_tokens=False)["input_ids"]) < CFG.passage_tokens:
            continue
        ids = ids[: CFG.passage_tokens]
        out.append({"index": len(out), "split": "train", "title": art["title"], "header_line": art["header_line"],
                    "token_ids": ids, "text": body[:6000],
                    "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        if len(out) == CFG.n_passages:
            break
    assert len(out) == CFG.n_passages
    for p in out:  # the stored text must still give >= 256 GPT-2 tokens
        assert len(g2(p["text"], add_special_tokens=False)["input_ids"]) >= CFG.passage_tokens
    seen = set()
    for f in ("wikitext_manifest.json", "wikitext_fresh_manifest.json", "wikitext_fresh2_manifest.json"):
        seen |= {json.dumps(p["token_ids"]) for p in json.loads((REPO / "stimuli" / f).read_text())["passages"]}
    assert not any(json.dumps(p["token_ids"]) in seen for p in out), "overlap with earlier passages"
    ids_all = [p["token_ids"] for p in out]
    m = {"dataset": "Salesforce/wikitext", "config": CFG.wikitext_config, "splits": ["train"], "rule": RULE,
         "passage_tokens": CFG.passage_tokens, "sha256_all": hashlib.sha256(json.dumps(ids_all).encode()).hexdigest(),
         "passages": out}
    path = REPO / "stimuli" / "wikitext_fresh3_manifest.json"
    path.write_text(json.dumps(m, indent=1))
    print(len(out), "passages ->", path.name, "| sha256_all", m["sha256_all"][:16], "| first:", out[0]["title"])


if __name__ == "__main__":
    main()
