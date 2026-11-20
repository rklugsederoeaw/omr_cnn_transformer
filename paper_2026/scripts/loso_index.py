"""Index einer Leave-one-source-out-Faltung aus der festen Aufteilung (trainingsdaten/omr_gt).

    python3 -I skripte/loso_index.py <name> <quelle> [<quelle> ...]
    python3 -I skripte/loso_index.py test     # Nachbau von daten/loso_a-kn-1012 (Selbsttest)

Regel wie beim ersten Lauf (A-KN 1012, 13.10.2026): Alle Zeilen der gesperrten Quellen fallen aus Training und
Validierung, auch ihre Ernte (`<quelle>.ernte*`). Testmenge sind alle Zeilen des Hauptexports der gesperrten
Quellen aus allen drei Teilen, jede Zeile einmal (die Übergewichtungskopien des Index fallen weg). Die Testzeilen
der übrigen Quellen bleiben draußen; Training und Validierung der übrigen Quellen bleiben unverändert.
Bildpfade relativ zu daten/loso_<name> (der Datenlader nimmt die Elternmappe von music/ als Basis), damit
der Baum auf dem Cluster unter derselben Struktur liegt.
Ausgabe: daten/loso_<name>/music/index.jsonl
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # natural_music_processing
ROOT = HERE.parent  # volpiano2omr_gt
OMR_GT = ROOT / "trainingsdaten/omr_gt"


def gesperrt(source, quellen):
    return any(source == q or source.startswith(q + ".") for q in quellen)


def bauen(quellen, basis_neu):
    alt = [json.loads(z) for z in (OMR_GT / "music/index.jsonl").open()]
    aus, gesehen = [], set()
    for x in alt:
        bild = (OMR_GT / x["image"]).resolve()
        x = dict(x, image=os.path.relpath(bild, basis_neu))
        if gesperrt(x["source"], quellen):
            if x["source"] in quellen and (x["source"], x["id"]) not in gesehen:
                gesehen.add((x["source"], x["id"]))
                aus.append(dict(x, split="test"))
        elif x["split"] != "test":
            aus.append(x)
    return aus


def schreiben(name, quellen):
    basis = HERE / "daten" / f"loso_{name}"
    zeilen = bauen(quellen, basis)
    (basis / "music").mkdir(parents=True, exist_ok=True)
    (basis / "music/index.jsonl").write_text(
        "".join(json.dumps(z, ensure_ascii=False) + "\n" for z in zeilen)
    )
    fehlt = [z["image"] for z in zeilen if not (basis / z["image"]).is_file()]
    assert not fehlt, fehlt[:3]
    print(name, Counter(z["split"] for z in zeilen))


def selbsttest():
    alt_basis = HERE / "daten/loso_a-kn-1012"
    neu = bauen(["a-kn-1012"], alt_basis)
    alt = [json.loads(z) for z in (alt_basis / "music/index.jsonl").open()]

    def k(z):
        return (
            z["split"],
            z["source"],
            z["id"],
            str((alt_basis / z["image"]).resolve()),
            z["label"],
            tuple(z["bbox"]),
        )

    assert Counter(map(k, neu)) == Counter(map(k, alt)), (
        Counter(z["split"] for z in neu),
        Counter(z["split"] for z in alt),
    )
    assert gesperrt("d-ka-aug-lx-addendum", ["d-ka-aug-lx-addendum"])
    assert not gesperrt("d-ka-aug-lx-addendum", ["d-ka-aug-lx"])
    print(
        "Selbsttest: Nachbau von loso_a-kn-1012 gleich",
        Counter(z["split"] for z in neu),
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        selbsttest()
    else:
        schreiben(sys.argv[1], sys.argv[2:])
