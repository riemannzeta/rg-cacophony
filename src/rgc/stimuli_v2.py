"""Build and validate the PREREG-2 part B (graded / conflicting cue) stimuli."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rgc.config import CFG
from rgc.stimuli import STOPWORDS, _tokenize_aligned, content_words, parse_phrase, stem
from rgc.stimuli_source import WORDS as V1
from rgc.stimuli_v2_source import CONDITIONS, CUES, FILLERS, FRAMES

STIM_V2_PATH = Path(__file__).resolve().parents[2] / "stimuli" / "conflict_v2.json"
H = {"AA": 2, "BB": -2, "AN": 1, "NA": 1, "BN": -1, "NB": -1, "AB": 0, "BA": 0, "NN": 0}


@dataclass
class PromptV2:
    id: str
    word: str
    split: str
    frame: int
    cond: str
    h: int
    slot1: str
    slot2: str
    prefix: str
    text: str
    control_text: str
    target_pos: int
    cueA_pos: list[int]
    cueB_pos: list[int]
    n_tokens: int
    token_ids: list[int]
    control_token_ids: list[int]


def _words(phrase: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", phrase) if w.lower() not in STOPWORDS]


def _assign(frame: int) -> tuple[int, int, int, int]:
    """Latin-square cue indices (a1, a2) for A and B; train frames use 0-3, test frames 4-7."""
    base, j = (0, frame) if frame < 8 else (4, frame - 8)
    return base + j % 4, base + (j + 1) % 4, j % 8, (j + 4) % 8


def build(tok) -> tuple[list[PromptV2], dict]:
    errors, warnings = [], []
    assert len(FRAMES) == 16 and len(FILLERS) == 8
    for f in FRAMES:
        assert f.count("{c1}") == 1 and f.count("{c2}") == 1 and f.endswith(" the"), f
    filler_vocab = {stem(w) for f in FILLERS for w in content_words(f)}
    all_v2_cues = set()
    prompts: list[PromptV2] = []
    for word, senses_spec in CUES.items():
        senses = V1[word]["senses"]
        assert tuple(senses_spec) == senses, (word, tuple(senses_spec), senses)
        v1_vocab = set()
        for split in ("train", "test"):
            for pair in V1[word][split]:
                for ph in pair:
                    v1_vocab |= {stem(w) for w in content_words(parse_phrase(ph)[0])}
        sense_vocab = {}
        for sense in senses:
            phrases = senses_spec[sense]
            assert len(phrases) == 8, (word, sense)
            voc_tr = {stem(w) for ph in phrases[:4] for w in content_words(ph)}
            voc_te = {stem(w) for ph in phrases[4:] for w in content_words(ph)}
            if voc_tr & voc_te:
                errors.append(f"{word}/{sense}: train/test cue overlap {sorted(voc_tr & voc_te)}")
            sense_vocab[sense] = voc_tr | voc_te
            all_v2_cues |= sense_vocab[sense]
            for ph in phrases:
                for w in content_words(ph):
                    sw = stem(w)
                    if sw == stem(word) or w.startswith(word):
                        errors.append(f"{word}/{sense}: cue phrase contains target: {ph!r}")
                    if sw in v1_vocab:
                        errors.append(f"{word}/{sense}: cue word {w!r} already used in v1 for this word")
                    near = [v for v in v1_vocab if v != sw and min(len(v), len(sw)) >= 4 and (v.startswith(sw) or sw.startswith(v))]
                    if near:
                        warnings.append(f"{word}/{sense}: {w!r} near-overlaps v1 {near}")
        shared = sense_vocab[senses[0]] & sense_vocab[senses[1]]
        if shared:
            errors.append(f"{word}: cue words shared by both senses {sorted(shared)}")
        for f in FILLERS:
            if any(stem(w) == stem(word) for w in content_words(f)):
                errors.append(f"filler {f!r} contains target {word}")

        for fi, frame in enumerate(FRAMES):
            i1, i2, n1, n2 = _assign(fi)
            slot = {"A": (senses_spec[senses[0]][i1], senses_spec[senses[0]][i2]),
                    "B": (senses_spec[senses[1]][i1], senses_spec[senses[1]][i2]),
                    "N": (FILLERS[n1], FILLERS[n2])}
            for cond in CONDITIONS:
                s1, s2 = slot[cond[0]][0], slot[cond[1]][1]
                prefix = frame.format(c1=s1, c2=s2)
                # cue char spans: content words of A/B cue phrases in their slot positions
                spans = {"A": [], "B": []}
                start1 = frame.index("{c1}")
                start2 = frame.format(c1=s1, c2="{c2}").index("{c2}")
                for which, phrase, start in ((cond[0], s1, start1), (cond[1], s2, start2)):
                    if which == "N":
                        continue
                    for m in re.finditer(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", phrase):
                        if m.group(0).lower() not in STOPWORDS:
                            spans[which].append((start + m.start(), start + m.end()))
                text, ids, tpos, cA = _tokenize_aligned(tok, prefix, word, "", spans["A"])
                _, _, _, cB = _tokenize_aligned(tok, prefix, word, "", spans["B"])
                ctext, cids, ctpos, _ = _tokenize_aligned(tok, prefix, CFG.control_token.strip(), "", [])
                assert ctpos == tpos and len(cids) == len(ids)
                prompts.append(PromptV2(
                    id=f"{word}/{'train' if fi < 8 else 'test'}/{fi:02d}/{cond}", word=word,
                    split="train" if fi < 8 else "test", frame=fi, cond=cond, h=H[cond], slot1=s1, slot2=s2,
                    prefix=prefix, text=text, control_text=ctext, target_pos=tpos,
                    cueA_pos=cA if "A" in cond else [], cueB_pos=cB if "B" in cond else [],
                    n_tokens=len(ids), token_ids=ids, control_token_ids=cids,
                ))
    leak = sorted(filler_vocab & all_v2_cues)
    if leak:
        errors.append(f"filler words used as cue words: {leak}")
    return prompts, {"errors": errors, "warnings": warnings, "n_prompts": len(prompts)}


def save(prompts, path: Path = STIM_V2_PATH) -> None:
    path.write_text(json.dumps([asdict(p) for p in prompts], indent=1))


def load(path: Path = STIM_V2_PATH) -> list[PromptV2]:
    return [PromptV2(**d) for d in json.loads(path.read_text())]
