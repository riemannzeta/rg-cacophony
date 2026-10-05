"""Build, validate and load the Measurement-1 stimulus set.

Positions in the built records are SEQUENCE positions (BOS = 0), matching rgc.models.tokenize.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rgc.config import CFG
from rgc.stimuli_source import CONTINUATIONS, NEUTRAL_FRAMES, TEST_FRAMES, TRAIN_FRAMES, WORDS

STIM_PATH = Path(__file__).resolve().parents[2] / "stimuli" / "ambiguous.json"

STOPWORDS = set(
    "a an the of on in at with to and for our my his her their its by from that this was were is "
    "all who we i he she they it had have be been not would there up out over into after about "
    "under near what which when where did".split()
)


@dataclass
class Prompt:
    id: str
    word: str
    sense: str  # sense label, or "neutral"
    split: str  # "train" | "test" | "neutral"
    frame: int  # index into the split's frame list
    pair: int  # minimal-pair index within split (= frame for train/test; -1 for neutral)
    continuation: int
    prefix: str  # text before the target (no brackets)
    cue_words: list[str]
    text: str  # prefix + " " + word + continuation
    control_text: str  # prefix + control token + continuation
    # Sequence positions (BOS = 0):
    target_pos: int
    cue_pos: list[int]
    n_tokens: int  # including BOS
    token_ids: list[int]
    control_token_ids: list[int]


def parse_phrase(phrase: str) -> tuple[str, list[str]]:
    return phrase.replace("[", "").replace("]", ""), re.findall(r"\[([^\]]+)\]", phrase)


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower()) if w not in STOPWORDS]


def stem(w: str) -> str:
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _cue_char_spans(prefix: str, frame: str, phrase: str) -> list[tuple[int, int]]:
    """Character spans (in prefix) of bracketed cue words."""
    start = frame.index("{c}")
    spans, offset, clean_pos = [], start, 0
    for m in re.finditer(r"\[([^\]]+)\]|[^\[]+", phrase):
        if m.group(1) is not None:
            spans.append((offset + clean_pos, offset + clean_pos + len(m.group(1))))
            clean_pos += len(m.group(1))
        else:
            clean_pos += len(m.group(0))
    for a, b in spans:
        assert prefix[a:b] in phrase, (prefix[a:b], phrase)
    return spans


def _tokenize_aligned(tok, prefix: str, target: str, continuation: str, cue_spans):
    text = f"{prefix} {target}{continuation}"
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    assert ids[: len(prefix_ids)] == prefix_ids, f"prefix tokenisation changes in context: {text!r}"
    tgt_ids = tok(" " + target, add_special_tokens=False)["input_ids"]
    assert len(tgt_ids) == 1, f"target {target!r} is not a single token"
    t_idx = len(prefix_ids)
    assert ids[t_idx] == tgt_ids[0], f"target token not at expected index: {text!r}"
    cont_ids = tok(continuation, add_special_tokens=False)["input_ids"]
    assert ids[t_idx + 1 :] == cont_ids, f"continuation tokenisation changes in context: {text!r}"
    cue_tok = sorted(
        {i for i, (a, b) in enumerate(offs[:t_idx]) for (ca, cb) in cue_spans if a < cb and b > ca}
    )
    assert len(cue_spans) == 0 or cue_tok, f"cue words not mapped to tokens: {text!r}"
    bos = [tok.bos_token_id] if CFG.prepend_bos else []
    shift = len(bos)
    return text, bos + ids, t_idx + shift, [i + shift for i in cue_tok]


def build(tok) -> tuple[list[Prompt], dict]:
    """Build all prompts and run every validation check. Returns (prompts, report)."""
    report: dict = {"words": {}, "errors": [], "warnings": []}
    ctrl = CFG.control_token
    ctrl_ids = tok(ctrl, add_special_tokens=False)["input_ids"]
    if len(ctrl_ids) != 1:
        report["errors"].append(f"control token {ctrl!r} is not a single token: {ctrl_ids}")
    assert len(TRAIN_FRAMES) == CFG.n_train_per_sense
    assert len(TEST_FRAMES) == CFG.n_per_sense - CFG.n_train_per_sense
    assert len(NEUTRAL_FRAMES) == CFG.n_neutral
    assert len(CONTINUATIONS) == CFG.n_continuations
    for f in TRAIN_FRAMES + TEST_FRAMES:
        assert f.count("{c}") == 1 and f.endswith(" the"), f
    for f in NEUTRAL_FRAMES:
        assert f.endswith(" the"), f

    prompts: list[Prompt] = []
    for word, spec in WORDS.items():
        senses = spec["senses"]
        w_ids = tok(" " + word, add_special_tokens=False)["input_ids"]
        info = {"token_id": w_ids, "single_token": len(w_ids) == 1, "senses": list(senses)}
        if len(w_ids) != 1:
            report["errors"].append(f"{word}: not a single token {w_ids}")
        assert len(spec["train"]) == CFG.n_train_per_sense, word
        assert len(spec["test"]) == CFG.n_per_sense - CFG.n_train_per_sense, word

        # ---- lexical checks -------------------------------------------------------------------
        vocab = {s: {sp: set() for sp in senses} for s in ("train", "test")}
        cues = {s: {sp: set() for sp in senses} for s in ("train", "test")}
        for split in ("train", "test"):
            for pair in spec[split]:
                for sense, phrase in zip(senses, pair):
                    clean, cw = parse_phrase(phrase)
                    vocab[split][sense] |= {stem(w) for w in content_words(clean)}
                    cues[split][sense] |= {stem(w) for c in cw for w in content_words(c)}
                    for w in content_words(clean):
                        if stem(w) == stem(word):
                            report["errors"].append(f"{word}: target word inside cue phrase {phrase!r}")
                        elif w.startswith(word) or word in w.split("-"):
                            report["warnings"].append(f"{word}: cue word {w!r} contains target")
        train_all = vocab["train"][senses[0]] | vocab["train"][senses[1]]
        test_all = vocab["test"][senses[0]] | vocab["test"][senses[1]]
        overlap = sorted(train_all & test_all)
        if overlap:
            report["errors"].append(f"{word}: content words shared by train and test cue phrases: {overlap}")
        near = sorted(
            (a, b) for a in train_all for b in test_all
            if a != b and min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))
        )
        if near:
            report["warnings"].append(f"{word}: near-overlap train/test (prefix) {near}")
        for split in ("train", "test"):
            both = sorted(cues[split][senses[0]] & cues[split][senses[1]])
            if both:
                report["errors"].append(f"{word}/{split}: cue word used for both senses: {both}")
        info["n_cue_words"] = {sp: {s: len(cues[sp][s]) for s in senses} for sp in ("train", "test")}

        # ---- build prompts --------------------------------------------------------------------
        for split, frames in (("train", TRAIN_FRAMES), ("test", TEST_FRAMES)):
            for i, (frame, pair) in enumerate(zip(frames, spec[split])):
                cont = CONTINUATIONS[i % len(CONTINUATIONS)]
                for sense, phrase in zip(senses, pair):
                    clean, cw = parse_phrase(phrase)
                    prefix = frame.format(c=clean)
                    spans = _cue_char_spans(prefix, frame, phrase)
                    text, ids, tpos, cpos = _tokenize_aligned(tok, prefix, word, cont, spans)
                    ctext, cids, ctpos, _ = _tokenize_aligned(tok, prefix, ctrl.strip(), cont, [])
                    assert ctpos == tpos and len(cids) == len(ids)
                    prompts.append(Prompt(
                        id=f"{word}/{split}/{sense}/{i:02d}", word=word, sense=sense, split=split,
                        frame=i, pair=i, continuation=i % len(CONTINUATIONS), prefix=prefix,
                        cue_words=cw, text=text, control_text=ctext, target_pos=tpos, cue_pos=cpos,
                        n_tokens=len(ids), token_ids=ids, control_token_ids=cids,
                    ))
        for j, frame in enumerate(NEUTRAL_FRAMES):
            cont = CONTINUATIONS[j % len(CONTINUATIONS)]
            text, ids, tpos, _ = _tokenize_aligned(tok, frame, word, cont, [])
            ctext, cids, _, _ = _tokenize_aligned(tok, frame, ctrl.strip(), cont, [])
            prompts.append(Prompt(
                id=f"{word}/neutral/{j:02d}", word=word, sense="neutral", split="neutral", frame=j,
                pair=-1, continuation=j % len(CONTINUATIONS), prefix=frame, cue_words=[], text=text,
                control_text=ctext, target_pos=tpos, cue_pos=[], n_tokens=len(ids), token_ids=ids,
                control_token_ids=cids,
            ))

        # ---- balance stats --------------------------------------------------------------------
        ps = [p for p in prompts if p.word == word and p.split in ("train", "test")]
        by = {s: [p for p in ps if p.sense == s] for s in senses}
        info["mean_target_pos"] = {s: sum(p.target_pos for p in by[s]) / len(by[s]) for s in senses}
        info["mean_n_cue_tokens"] = {s: sum(len(p.cue_pos) for p in by[s]) / len(by[s]) for s in senses}
        diffs = [abs(a.target_pos - b.target_pos) for a, b in zip(by[senses[0]], by[senses[1]])]
        info["pair_target_pos_absdiff_mean"] = sum(diffs) / len(diffs)
        report["words"][word] = info

    report["n_prompts"] = len(prompts)
    report["continuation_tokens"] = [len(tok(c, add_special_tokens=False)["input_ids"]) for c in CONTINUATIONS]
    report["control_token"] = {"text": ctrl, "ids": ctrl_ids}
    return prompts, report


def save(prompts: list[Prompt], path: Path = STIM_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in prompts], indent=1))


def load(path: Path = STIM_PATH) -> list[Prompt]:
    return [Prompt(**d) for d in json.loads(path.read_text())]
