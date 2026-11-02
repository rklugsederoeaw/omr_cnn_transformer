"""CantusDB concordance scoring used by the original OMR reading chain."""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process

GAMUT = "9abcdefghjklmnopqrs"
CLEF_NOTE = {"c": GAMUT.index("k"), "f": GAMUT.index("f")}
POS = re.compile(r"^[lz]-?\d+$")
CLEF = re.compile(r"[a-g]-?\d(/[a-g]\d)*")
EXCLUDED_SOURCES = {123661}


def pos_num(name: str) -> int:
    k = int(name[1:])
    if name[0] == "l":
        return 2 * (k - 1) if k > 0 else 2 * k
    return 2 * k - 1 if k > 0 else 2 * k + 1


def collapse(value: str) -> str:
    return "".join(c for i, c in enumerate(value) if i == 0 or c != value[i - 1])


def pitches(tokens: list[str]) -> str:
    out, clef = [], None
    for token in tokens:
        if CLEF.fullmatch(token):
            effective = next((part for part in token.split("/") if part[0] in "cf"), None)
            if effective:
                clef = (CLEF_NOTE[effective[0]], 2 * (int(effective[1:]) - 1))
        elif POS.fullmatch(token) and clef:
            index = pos_num(token) - clef[1] + clef[0]
            if 0 <= index < len(GAMUT):
                out.append(GAMUT[index])
    return collapse("".join(out)) if clef else ""


def folio_key(folio: str):
    return [(0, int(piece), "") if piece.isdigit() else (1, 0, piece) for piece in re.findall(r"\d+|\D+", folio or "")]


def versions(db_path: Path, source_id: int, folio: str) -> list[str]:
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        chants = db.execute("SELECT folio, sequence, cantus_id FROM chants WHERE source_id=? AND cantus_id IS NOT NULL", (source_id,)).fetchall()
        chants.sort(key=lambda row: (folio_key(row[0]), int(row[1]) if str(row[1] or "").isdigit() else 0))
        positions = [i for i, row in enumerate(chants) if row[0] == folio]
        ids = [chants[i][2] for i in positions]
        if positions and positions[0] > 0:
            ids.insert(0, chants[positions[0] - 1][2])
        seen = set()
        for cantus_id in dict.fromkeys(ids):
            for sid, melody in db.execute("SELECT source_id, volpiano FROM chants WHERE cantus_id=? AND volpiano IS NOT NULL", (cantus_id,)):
                tones = "".join(c for c in melody.split("4")[0].lower() if c in GAMUT)
                if sid != source_id and sid not in EXCLUDED_SOURCES and len(tones) >= 4:
                    seen.add(collapse(tones))
        return sorted(seen)
    finally:
        db.close()


def shift(melody: str, steps: int) -> str:
    return "".join(GAMUT[GAMUT.index(p) + steps] for p in melody if 0 <= GAMUT.index(p) + steps < len(GAMUT))


def score(candidates: list[list[str]], reference: int, source_id: int, folio: str, db_path: Path) -> list[float]:
    variants = versions(db_path, source_id, folio)
    reference_pitch = pitches(candidates[reference])
    majority = 0
    if reference_pitch and variants:
        votes = Counter()
        for variant in variants:
            best = max(range(-7, 8), key=lambda step: (fuzz.partial_ratio(shift(reference_pitch, step), variant), -abs(step)))
            votes[best] += 1
        majority = min(votes, key=lambda step: (-votes[step], abs(step), step))
    scores = []
    for candidate in candidates:
        pitch = pitches(candidate)
        if not pitch or not variants:
            scores.append(0.0)
            continue
        match = process.extractOne(pitch, variants, scorer=fuzz.partial_ratio)
        rag = match[1] / 100 * sum(bool(POS.fullmatch(token)) for token in candidate)
        offset = GAMUT.index(pitch[0]) - GAMUT.index(reference_pitch[0]) if reference_pitch else 0
        scores.append(0.0 if offset != 0 and offset != majority else rag)
    return scores
