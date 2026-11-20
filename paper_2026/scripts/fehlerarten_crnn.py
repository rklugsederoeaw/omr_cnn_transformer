"""Fehlerarten der CRNN-Lesung gegen die Referenz (Artikel, Abschnitt 6.5).

    .venv_helsen/bin/python -I skripte/fehlerarten_crnn.py <index.jsonl> <lesungen_voll.jsonl> [FELD] [...]
    .venv_helsen/bin/python -I skripte/fehlerarten_crnn.py test

Je Paar aus Index (split test) und Lesung; FELD = ohne_vorwissen (Vorgabe, gemeinsame Decodierung) oder gelesen.
Die Editieroperationen der Tokenfolgen (Levenshtein) werden nach Art gezählt: Note fehlt / überzählig / Lage falsch um
eine Stufe / um mehr; Schlüssel, Trenner (`,`), Abschnittsstrich (`L`), Vorzeichen (`b_`, `na_`), je ersetzt, fehlend
oder überzählig zusammengefasst. Tonwiederholung: Anteil der fehlenden Noten, deren Lage gleich der vorigen Note der
Referenz ist. Ausgabe je Paar: Token-SER, Anteile der Fehlerarten an allen Fehlern, Anteil Wiederholungen; je Quelle
die Token-SER (nur bei mehreren Quellen).
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein as L

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auswertung_vorwissen import key_of  # noqa: E402

POS = re.compile(r"^[lz]-?\d+$")


def stufe(t):
    """Lage als Zahl (l1 = 0, z1 = 1, l2 = 2, ..., z-1 = -1, l-1 = -2)."""
    n = int(t[1:])
    if t[0] == "l":
        return 2 * (n - 1) if n > 0 else 2 * n
    return 2 * n - 1 if n > 0 else 2 * n + 1


def art(t):
    if POS.match(t):
        return "note"
    if t.startswith(("b_", "na_")):
        return "vorzeichen"
    if t == ",":
        return "trenner"
    if t == "L":
        return "strich"
    return "schluessel"


def zaehlen(ref, hyp, c):
    for op in L.editops(hyp, ref):  # Quelle = Lesung, Ziel = Referenz
        if op.tag == "insert":  # in der Lesung fehlt ein Referenztoken
            t = ref[op.dest_pos]
            a = art(t)
            if a == "note":
                c["Note fehlt"] += 1
                vorher = [x for x in ref[: op.dest_pos] if POS.match(x)]
                if vorher and vorher[-1] == t:
                    c["davon Tonwiederholung"] += 1
            else:
                c[a] += 1
        elif op.tag == "delete":
            t = hyp[op.src_pos]
            a = art(t)
            c["Note überzählig" if a == "note" else a] += 1
        else:
            r, h = ref[op.dest_pos], hyp[op.src_pos]
            if art(r) == art(h) == "note":
                d = abs(stufe(r) - stufe(h))
                c["Lage falsch um 1" if d == 1 else "Lage falsch um ≥2"] += 1
            else:
                c[art(r) if art(r) != "note" else art(h)] += 1


def auswerten(index, lesungen, feld):
    ref = {}
    for z in map(json.loads, Path(index).open()):
        if z.get("split") == "test":
            ref.setdefault(key_of(z["image"]), (z["label"].split(), z["source"]))
    c, je = Counter(), defaultdict(lambda: [0, 0])
    n = 0
    for x in map(json.loads, Path(lesungen).open()):
        if x["key"] not in ref:
            continue
        r, q = ref[x["key"]]
        h = x[feld]
        zaehlen(r, h, c)
        d = L.distance(h, r)
        je[q][0] += d
        je[q][1] += len(r)
        n += len(r)
    fehler = sum(v for k, v in c.items() if k != "davon Tonwiederholung")
    arten = [
        "Note fehlt",
        "Note überzählig",
        "Lage falsch um 1",
        "Lage falsch um ≥2",
        "schluessel",
        "trenner",
        "strich",
        "vorzeichen",
    ]
    print(
        f"{lesungen} [{feld}]: Token-SER {100 * sum(v[0] for v in je.values()) / n:.2f} %, {fehler} Fehler"
    )
    print("  " + "; ".join(f"{a} {100 * c[a] / fehler:.1f} %" for a in arten))
    print(
        f"  fehlende Noten, die eine Tonwiederholung sind: {100 * c['davon Tonwiederholung'] / max(c['Note fehlt'], 1):.1f} %"
    )
    if len(je) > 1:
        print(
            "  je Quelle: "
            + "; ".join(f"{q} {100 * a / b:.2f}" for q, (a, b) in sorted(je.items()))
        )
    return c


def selbsttest():
    c = Counter()
    zaehlen(
        "c3 l2 l2 , z2 L".split(), "c3 l2 , l3 L".split(), c
    )  # z2 -> l3: eine Stufe
    assert (
        c["Note fehlt"] == 1
        and c["davon Tonwiederholung"] == 1
        and c["Lage falsch um 1"] == 1
    ), c
    c = Counter()
    zaehlen("c3 l2".split(), "f3 l2 l4".split(), c)
    assert c["schluessel"] == 1 and c["Note überzählig"] == 1, c
    assert (
        stufe("l1") == 0
        and stufe("z1") == 1
        and stufe("z-1") == -1
        and stufe("l-1") == -2
    )
    print("Selbsttest in Ordnung")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["test"]:
        selbsttest()
    else:
        i = 0
        while i < len(a):
            feld = (
                a[i + 2]
                if i + 2 < len(a) and a[i + 2] in ("ohne_vorwissen", "gelesen")
                else "ohne_vorwissen"
            )
            auswerten(a[i], a[i + 1], feld)
            i += (
                3 if i + 2 < len(a) and a[i + 2] in ("ohne_vorwissen", "gelesen") else 2
            )
