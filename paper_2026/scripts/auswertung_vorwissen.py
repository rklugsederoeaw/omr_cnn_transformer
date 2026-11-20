"""Wirkung der Stufen der Lesekette je Testmenge: Token- und Melodie-SER, gepaarter Bootstrap, Schlüsseländerungen.

    .venv_helsen/bin/python -I skripte/auswertung_vorwissen.py <index.jsonl> <lesen_rag-Ordner> [...]
    .venv_helsen/bin/python -I skripte/auswertung_vorwissen.py test

Gelesen werden lesungen_{voll,nur_lm,nur_konk}.jsonl (lesen_reihe.py); Referenz sind die Labels des Index (split test),
verbunden über den Schlüssel <Ordner über images>/<Bildname> wie in lesen_rag.py. Maße wie in rag_bewertung.py:
Token-SER = Levenshtein der Tokenfolgen; Melodie-SER = Levenshtein der Tonhöhen über den gelesenen Schlüssel
(Wiederholungen zusammengezogen), nur Zeilen mit Anfangsschlüssel in der Referenz. Bootstrap über Zeilen und, wenn
GRUPPE=folio gesetzt ist, über Paare aus Quelle und Folio (Folio = Teil der Zeilen-id vor dem ersten `_`; Quellen-
suffixe wie `.ernte` zählen zur Handschrift; Berichtigung nach Gutachten codex M1, 20.11.2026: vorher nur die Folionummer,
wodurch gleich nummerierte Folios verschiedener Handschriften eine Gruppe bildeten), je 2.000 Ziehungen, 95 %.
Schlüssel: Anteil der Zeilen mit richtigem Anfangsschlüssel; bei Zeilen, in denen eine Stufe den Anfangsschlüssel
gegenüber der Lesung ohne Vorwissen ändert, wie oft er dadurch richtig bzw. falsch wird.
"""

import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein as L

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_bewertung import CLEF_NOTE, pitches  # noqa: E402

SCHL = re.compile(r"[a-g]-?\d(/[a-g]\d)*$")


def key_of(image):
    p = Path(image)
    return f"{p.parent.parent.name}/{p.stem}"


def folio_gruppe(source, zid):
    """(Handschrift, Folio): Ernte-Suffixe gehören zur Handschrift, das Folio ist der Teil der id vor dem ersten `_`."""
    return source.split(".")[0] + "/" + zid.split("_")[0]


def schluessel(t):
    """Wirksamer Anfangsschlüssel als Tonstufe der untersten Linie (wie rag_bewertung.pitches): c4 und f2 sind gleich,
    bei zusammengesetzten Schlüsseln (f1/c3) zählt der erste C- oder F-Teil; None ohne Anfangsschlüssel."""
    if not t or not SCHL.match(t[0]):
        return None
    eff = next((x for x in t[0].split("/") if x[0] in "cf"), None)
    return CLEF_NOTE[eff[0]] - 2 * (int(eff[1:]) - 1) if eff else None


def zeile(ref, hyp):
    """(Token-Distanz, Tokenlänge, Melodie-Distanz, Melodielänge oder 0, Schlüssel richtig)"""
    pr, ph = pitches(ref), pitches(hyp)
    m = (L.distance(ph or "", pr), len(pr)) if pr is not None else (0, 0)
    return (L.distance(hyp, ref), len(ref), *m, schluessel(hyp) == schluessel(ref))


def rate(f, ids, k):
    n = sum(f[i][k + 1] for i in ids)
    return 100 * sum(f[i][k] for i in ids) / max(n, 1)


def bootstrap(a, b, gruppen, k, n=2000):
    rnd = random.Random(0)
    g = list(gruppen.values())
    d = []
    for _ in range(n):
        ids = [i for x in (rnd.choice(g) for _ in g) for i in x]
        d.append(rate(a, ids, k) - rate(b, ids, k))
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]


def auswerten(index, ordner):
    ref = {}
    for z in map(json.loads, Path(index).open()):
        if z.get("split") == "test":
            ref.setdefault(
                key_of(z["image"]), (z["label"].split(), z["id"], z["source"])
            )
    stufen = {}
    for s in ("voll", "nur_lm", "nur_konk"):
        stufen[s] = {
            x["key"]: x
            for x in map(json.loads, (Path(ordner) / f"lesungen_{s}.jsonl").open())
        }
    ids = sorted(set(ref) & set(stufen["voll"]))
    assert len(ids) >= 0.99 * len(ref), (len(ids), len(ref))
    gruppe = os.environ.get("GRUPPE", "")
    gruppen = defaultdict(list)
    for i in ids:
        gruppen[folio_gruppe(ref[i][2], ref[i][1]) if gruppe == "folio" else i].append(
            i
        )
    les = {"gemeinsam": {i: stufen["voll"][i]["ohne_vorwissen"] for i in ids}}
    les.update(
        {
            f"+{s}": {i: stufen[s][i]["gelesen"] for i in ids}
            for s in ("nur_lm", "nur_konk", "voll")
        }
    )
    f = {n: {i: zeile(ref[i][0], h[i]) for i in ids} for n, h in les.items()}
    print(f"{ordner}: {len(ids)} Zeilen, {len(gruppen)} Gruppen ({gruppe or 'Zeile'})")
    for n, x in f.items():
        sk = 100 * sum(x[i][4] for i in ids) / len(ids)
        print(
            f"  {n:12s} Token {rate(x, ids, 0):6.2f}  Melodie {rate(x, ids, 2):6.2f}  Schlüssel {sk:5.1f} %"
        )
    for n in ("+nur_lm", "+nur_konk", "+voll"):
        t = bootstrap(f[n], f["gemeinsam"], gruppen, 0)
        m = bootstrap(f[n], f["gemeinsam"], gruppen, 2)
        geaendert = [
            i for i in ids if schluessel(les[n][i]) != schluessel(les["gemeinsam"][i])
        ]
        rich = sum(f[n][i][4] and not f["gemeinsam"][i][4] for i in geaendert)
        falsch = sum(f["gemeinsam"][i][4] and not f[n][i][4] for i in geaendert)
        print(
            f"  {n} − gemeinsam: Token {rate(f[n], ids, 0) - rate(f['gemeinsam'], ids, 0):+.2f} [{t[0]:+.2f}; {t[1]:+.2f}]"
            f"  Melodie {rate(f[n], ids, 2) - rate(f['gemeinsam'], ids, 2):+.2f} [{m[0]:+.2f}; {m[1]:+.2f}]"
            f"  Schlüssel geändert {len(geaendert)}: richtig geworden {rich}, falsch geworden {falsch}"
        )


def selbsttest():
    r = "c4 l2 , z2 l3".split()
    z = zeile(r, "c4 l2 , z2 l3".split())
    assert z[0] == 0 and z[2] == 0 and z[4] is True, z
    z = zeile(
        r, "c3 l2 , z2 l3".split()
    )  # anderer Schlüssel: Token 1, alle Töne verschoben
    assert z[0] == 1 and z[2] == 3 and z[4] is False, z
    assert (
        folio_gruppe("ch-p-18.ernte", "55_03")
        == folio_gruppe("ch-p-18", "55_01")
        != folio_gruppe("sbe-0611", "55_01")
    )
    assert schluessel(["c4"]) == schluessel(["f2"]) == schluessel(["d1/f2/c4"])
    assert schluessel(["c4"]) != schluessel(["c3"]) and schluessel([","]) is None
    a = {"x": (1, 10, 0, 5, True), "y": (0, 10, 0, 5, True)}
    assert rate(a, ["x", "y"], 0) == 5.0
    lo, hi = bootstrap(a, a, {"x": ["x"], "y": ["y"]}, 0)
    assert lo == hi == 0
    print("Selbsttest in Ordnung")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        selbsttest()
    else:
        args = sys.argv[1:]
        for index, ordner in zip(args[::2], args[1::2]):
            auswerten(index, ordner)
