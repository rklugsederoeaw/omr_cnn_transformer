"""Vergleicht Lesungen neuer Handschriften mit dem geprüften GT: Tonlage und Anfangsschlüssel.

GT-Lage (l_n Linie n, z_n Zwischenraum über l_n, z-1 unter l1, l-1 erste Hilfslinie unten)
wird auf die gabc-Buchstaben abgebildet. Ein Versatztest prüft, ob die Abbildung stimmt.

    python vergleich_gt.py daten/gt/lesungen.jsonl
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein as lev

from metriken import sicht

LAGE = {
    "z-2": "a",
    "l-1": "b",
    "z-1": "c",
    "l1": "d",
    "z1": "e",
    "l2": "f",
    "z2": "g",
    "l3": "h",
    "z3": "i",
    "l4": "j",
    "z4": "k",
    "l5": "l",
    "z5": "m",
}
ANDERE_LAGE = re.compile(r"^[lz]-?\d$")


def toene(label):
    """Tonfolge des GT in gabc-Buchstaben; Lagen außerhalb a-m werden zu ?, zählen also als Fehler."""
    return [
        LAGE.get(t, "?") for t in label.split() if t in LAGE or ANDERE_LAGE.match(t)
    ]


def anfangsschluessel(label):
    t = label.split()
    return t[0].split("/") if t and re.match(r"^[a-g]\d(/[a-g]\d)*$", t[0]) else None


def versetzt(folge, k):
    return [
        chr(ord(c) + k) if c != "?" and "a" <= chr(ord(c) + k) <= "m" else "?"
        for c in folge
    ]


def main(pfad):
    zeilen = [
        json.loads(z) for z in Path(pfad).read_text(encoding="utf-8").splitlines()
    ]
    je = defaultdict(list)
    for z in zeilen:
        je[z["quelle"]].append(z)
    je["alle"] = zeilen
    ergebnis = {}
    for q, rs in je.items():
        fehler = n = ok_start = n_start = gleich = n_hyp = 0
        versatz = defaultdict(int)
        for z in rs:
            r, h = toene(z["label"]), sicht(z["lesung"])["toene"]
            fehler += lev.distance(r, h)
            n += len(r)
            n_hyp += len(h)
            gleich += r == h
            for k in range(-4, 5):
                versatz[k] += lev.distance(r, versetzt(h, k))
            s = anfangsschluessel(z["label"])
            if s:
                n_start += 1
                ok_start += sicht(z["lesung"])["schl"][:1] in [[x] for x in s]
        ergebnis[q] = {
            "systeme": len(rs),
            "toene_gt": n,
            "toene_lesung": n_hyp,
            "tonlage_ser_pct": round(100 * fehler / n, 2),
            "systeme_fehlerfrei": gleich,
            "anfangsschluessel_richtig_pct": round(100 * ok_start / n_start, 1)
            if n_start
            else None,
            "abgebrochen": sum(z["abgebrochen"] for z in rs),
            "schleife": sum(z["schleife"] for z in rs),
            "versatz_ser_pct": {k: round(100 * v / n, 1) for k, v in versatz.items()},
        }
        print(q, json.dumps(ergebnis[q], ensure_ascii=False))
    Path(pfad).with_suffix(".vergleich.json").write_text(
        json.dumps(ergebnis, ensure_ascii=False, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    assert toene("f3 l2 , l2 z1 L l3 b_z3 z-1 l-1 l6") == list("ffehcb?")
    assert (
        anfangsschluessel("f2/c4 l2") == ["f2", "c4"]
        and anfangsschluessel("l2") is None
    )
    assert versetzt(["d", "?"], 1) == ["e", "?"] and versetzt(["m"], 1) == ["?"]
    if len(sys.argv) > 1:
        main(sys.argv[1])
