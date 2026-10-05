"""Deterministic WikiText-103 passage selection for Measurement 2.

Rule (fixed before any measurement): walk the wikitext-103-raw-v1 TEST split in order, then the
VALIDATION split (the test split has only 62 articles; see DECISIONS.md D9). An article starts
at a line of the form " = Title = " (exactly one '=' on each side). Each article body (all lines until the next article header, including its section headers) is detokenised with
the standard WikiText detokeniser (as in lm-evaluation-harness) and tokenised with the Pythia
tokenizer. The first `n_passages` articles whose body has at least `passage_tokens` tokens are
kept; each passage is the first `passage_tokens` tokens of its body. BOS is prepended at run time.
"""

import hashlib
import json
import re
from pathlib import Path

from rgc.config import CFG

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "stimuli" / "wikitext_manifest.json"
_HEADER = re.compile(r"^ = ([^=].*[^=]) = $")


def detokenize(s: str) -> str:
    """WikiText detokeniser from lm-evaluation-harness (EleutherAI)."""
    s = s.replace("s '", "s'")
    s = re.sub(r"/' [0-9]/", r"/'[0-9]/", s)
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = s.replace(" : ", ": ").replace(" ; ", "; ").replace(" . ", ". ").replace(" ! ", "! ")
    s = s.replace(" ? ", "? ").replace(" , ", ", ")
    s = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", s)
    s = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", s)
    s = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", s)
    s = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', s)
    s = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", s)
    s = s.replace("= = = =", "====").replace("= = =", "===").replace("= =", "==")
    s = s.replace(" " + chr(176) + " ", chr(176)).replace(" \n", "\n").replace("\n ", "\n")
    s = s.replace(" N ", " 1 ").replace(" 's", "'s")
    return s


def articles(lines: list[str]) -> list[dict]:
    out, cur = [], None
    for i, line in enumerate(lines):
        m = _HEADER.match(line.rstrip("\n"))
        if m:
            if cur:
                out.append(cur)
            cur = {"title": m.group(1).strip(), "header_line": i, "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur:
        out.append(cur)
    return out


def build_manifest(tok) -> dict:
    from datasets import load_dataset

    passages, fingerprints, n_lines = [], {}, {}
    for split in CFG.wikitext_splits:
        ds = load_dataset("Salesforce/wikitext", CFG.wikitext_config, split=split)
        fingerprints[split], n_lines[split] = getattr(ds, "_fingerprint", None), len(ds)
        _collect(tok, split, ds["text"], passages)
        if len(passages) == CFG.n_passages:
            break
    assert len(passages) == CFG.n_passages
    all_ids = [p["token_ids"] for p in passages]
    return {
        "dataset": "Salesforce/wikitext",
        "config": CFG.wikitext_config,
        "splits": list(CFG.wikitext_splits),
        "dataset_fingerprints": fingerprints,
        "n_lines": n_lines,
        "rule": __doc__.strip(),
        "passage_tokens": CFG.passage_tokens,
        "sha256_all": hashlib.sha256(json.dumps(all_ids).encode()).hexdigest(),
        "passages": passages,
    }


def _collect(tok, split: str, lines: list[str], passages: list[dict]) -> None:
    for art in articles(lines):
        body = detokenize("".join(art["lines"])).strip()
        ids = tok(body, add_special_tokens=False)["input_ids"]
        if len(ids) < CFG.passage_tokens:
            continue
        ids = ids[: CFG.passage_tokens]
        passages.append({
            "index": len(passages),
            "split": split,
            "title": art["title"],
            "header_line": art["header_line"],
            "n_body_lines": len(art["lines"]),
            "token_ids": ids,
            "text_preview": tok.decode(ids[:40]),
            "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        })
        if len(passages) == CFG.n_passages:
            return


def load_passages(path: Path = MANIFEST_PATH) -> list[list[int]]:
    m = json.loads(path.read_text())
    ids = [p["token_ids"] for p in m["passages"]]
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == m["sha256_all"]
    return ids
