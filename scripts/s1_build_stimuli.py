"""Stage 1: build and validate all stimuli (no model forward passes).

Writes:
  stimuli/ambiguous.json           M1 prompts (train/test/neutral, control-token versions)
  stimuli/wikitext_manifest.json   M2 passages (token ids, titles, hashes)
  results/stage1/stimuli_report.json
"""

import json

from transformers import AutoTokenizer

from rgc import stimuli, wikitext
from rgc.config import CFG
from rgc.determinism import REPO, env_info
from rgc.stimuli_source import DROPPED


def main() -> None:
    tok = AutoTokenizer.from_pretrained(CFG.hf_repo, revision=f"step{CFG.final_step}")
    # The tokenizer is identical across checkpoints; verify for the ones we use.
    for step in CFG.checkpoints:
        other = AutoTokenizer.from_pretrained(CFG.hf_repo, revision=f"step{step}")
        assert other.get_vocab() == tok.get_vocab(), f"tokenizer differs at step{step}"

    candidates = {}
    for w in list(CFG.candidate_words) + ["palm"]:
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        candidates[w] = {"ids": ids, "pieces": [tok.decode([i]) for i in ids], "single": len(ids) == 1}

    prompts, report = stimuli.build(tok)
    if report["errors"]:
        raise SystemExit("stimulus validation failed:\n" + "\n".join(report["errors"]))
    stimuli.save(prompts)

    manifest = wikitext.build_manifest(tok)
    wikitext.MANIFEST_PATH.write_text(json.dumps(manifest, indent=1))

    out = {
        "env": env_info(),
        "candidate_tokenisation": candidates,
        "dropped": DROPPED,
        "stimuli": report,
        "counts": {
            split: sum(p.split == split for p in prompts) for split in ("train", "test", "neutral")
        },
        "wikitext": {
            "n_passages": len(manifest["passages"]),
            "sha256_all": manifest["sha256_all"],
            "titles": [p["title"] for p in manifest["passages"]],
        },
    }
    path = REPO / "results" / "stage1" / "stimuli_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("counts", "dropped")}, indent=1))
    print("warnings:", report["warnings"])
    print("wikitext:", out["wikitext"]["n_passages"], "passages; first titles:", out["wikitext"]["titles"][:6])


if __name__ == "__main__":
    main()
