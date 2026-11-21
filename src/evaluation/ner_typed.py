"""
Typed entity-level NER scoring, shared by both arms.

The thesis scored NER with a lenient instrument: per-sentence sets of
lowercased entity surfaces with the entity type dropped (evaluate.py's
evaluate_ner for the fine-tuned arm, notebooks/03 for the API arm). This
module keeps the type. The fine-tuned arm emits BIO tag ids over the input
tokens and the API arm emits JSON keyed by type, so the common unit is the
(type, surface) pair, and typed_micro_prf micro-averages exact matches over
the corpus. For the fine-tuned arm, whose output is token-aligned, seqeval's
span F1 can be computed from bio_tags as a cross-check.

Pure Python: no torch, so it can be imported by analysis scripts and tests.
"""

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NER_ID2TAG = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC",
}
ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")

Entity = Tuple[str, str]  # (type, surface)


def bio_tags(tag_string: str) -> Optional[List[str]]:
    """'[5, 0, 6, ...]' -> ['B-LOC', 'O', 'I-LOC', ...]; None if no tag ids were generated."""
    ints = re.findall(r"\d+", tag_string)
    if not ints:
        return None
    return [NER_ID2TAG.get(int(t), "O") for t in ints]


def bio_to_entities(tokens: Sequence[str], tags: Sequence[str]) -> List[Entity]:
    """
    BIO-decode tags over tokens into (type, surface) pairs.

    Same span rules as evaluate.py's _decode_tag_string_to_entities and the
    notebook's get_true_entities: a span opens on B-, an I- continues an open
    span (keeping the B- tag's type), anything else closes it, and an I- with
    no open span is treated as O. zip() stops at the shorter sequence.
    """
    entities, current, current_type = [], [], None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current:
                entities.append((current_type, " ".join(current)))
            current, current_type = [tok], tag[2:]
        elif tag.startswith("I-") and current:
            current.append(tok)
        else:
            if current:
                entities.append((current_type, " ".join(current)))
            current, current_type = [], None
    if current:
        entities.append((current_type, " ".join(current)))
    return entities


def parse_api_entities(response_text: str) -> Optional[List[Entity]]:
    """
    An API arm JSON response -> (type, surface) pairs, or None if unparseable.
    Strips ```json fences exactly as notebooks/03's parse_entity_response does.
    """
    try:
        parsed = json.loads(re.sub(r"```json|```", "", str(response_text)).strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return [(etype, str(e)) for etype in ENTITY_TYPES for e in parsed.get(etype, []) or []]


def _key(entity: Entity, typed: bool, case_sensitive: bool) -> Tuple:
    etype, surface = entity
    surface = " ".join(surface.split())
    if not case_sensitive:
        surface = surface.lower()
    return (etype, surface) if typed else (surface,)


def micro_prf(
    gold: Iterable[Sequence[Entity]],
    pred: Iterable[Sequence[Entity]],
    typed: bool = True,
    case_sensitive: bool = False,
) -> Dict[str, float]:
    """
    Corpus-level micro precision/recall/F1 over per-sentence entity lists.
    Duplicates count (multiset matching), as in span-level CoNLL scoring.
    """
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        gc = Counter(_key(e, typed, case_sensitive) for e in g)
        pc = Counter(_key(e, typed, case_sensitive) for e in p)
        hit = sum((gc & pc).values())
        tp += hit
        fp += sum(pc.values()) - hit
        fn += sum(gc.values()) - hit
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def lenient_sentence_f1(gold: Sequence[Entity], pred: Sequence[Entity]) -> float:
    """The thesis instrument: set F1 over lowercased surfaces, types dropped, 1.0 if both empty."""
    g = {s.lower().strip() for _, s in gold}
    p = {s.lower().strip() for _, s in pred}
    if not g and not p:
        return 1.0
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
