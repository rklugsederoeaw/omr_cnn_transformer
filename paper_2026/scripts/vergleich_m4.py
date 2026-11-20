"""Tonlagen-SER und Anfangsschlüssel für CRNN- und LightOn-Lesungen auf derselben Referenz (Messreihen M3, M4).

    .venv_helsen/bin/python -I skripte/vergleich_m4.py <referenz.jsonl> NAME=DATEI[:FELD] [NAME=DATEI[:FELD] ...]
    .venv_helsen/bin/python -I skripte/vergleich_m4.py test

Referenz: JSONL mit id, quelle, optional alias (id derselben Zeile bei LightOn) und label (CRNN-Token, M4) oder ziel (gabc, M3). Lesungen: JSONL mit id oder key
(lesen_rag: key = <quelle>/<id>) und dem Lesefeld (Vorgabe: gelesen bei CRNN-Lesungen, lesung bei LightOn); eine
Liste von Token oder ein Text. Gabc wird über metriken.sicht in Töne zerlegt, CRNN-Token über vergleich_gt.toene,
also dasselbe Maß wie im LightOn-Artikel (Tonlagen-SER = Levenshtein-Distanz der Tonfolgen / Töne der Referenz).
Ausgabe: je Lesung und Quelle Tonlagen-SER und Anfangsschlüssel richtig; für jedes Paar von Lesungen der Unterschied
mit gepaartem Bootstrap über Systeme (2.000 Ziehungen, 95 %); mit GRUPPE=seite über ganze Seiten (Folios).
"""

import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein as lev

sys.path.insert(
    0, os.environ.get("LIGHTON_CODE", "/home/rklug/projects_2026/lighton_choral")
)
from metriken import sicht  # noqa: E402
from vergleich_gt import anfangsschluessel, toene  # noqa: E402

GABC = re.compile(r"\(")


def folge(x):
    """-> (Tonfolge in gabc-Buchstaben, Anfangsschlüssel oder None) aus CRNN-Token oder gabc."""
    if isinstance(x, list):
        x = " ".join(x)
    if GABC.search(x):
        v = sicht(x)
        return v["toene"], (v["schl"][0] if v["schl"] else None)
    s = anfangsschluessel(x)
    return toene(x), (s[0] if s else None)


def referenz(z):
    if "label" in z:
        return toene(z["label"]), anfangsschluessel(z["label"])
    t, s = folge(z["ziel"])
    return t, [s] if s else None


def lesungen(arg, ids, alias):
    name, _, rest = arg.partition("=")
    pfad, _, feld = rest.partition(":")
    aus = {}
    for z in map(json.loads, Path(pfad).open(encoding="utf-8")):
        i = z.get("id") or z["key"]
        i = alias.get(i, i)  # LightOn-id -> Referenz-id
        feld_ = feld or ("gelesen" if "gelesen" in z else "lesung")
        aus[i] = folge(z[feld_])
    fehlt = ids - aus.keys()
    assert not fehlt, (
        f"{name}: {len(fehlt)} Systeme ohne Lesung, z. B. {sorted(fehlt)[:3]}"
    )
    return name, aus


def fehler(ref, hyp):
    """je System: (Distanz, Referenztöne, Schlüssel richtig oder None)"""
    return {
        i: (
            lev.distance(r[0], hyp[i][0]),
            len(r[0]),
            None if not r[1] else hyp[i][1] in r[1],
        )
        for i, r in ref.items()
    }


def gruppe(i):
    """GRUPPE=seite: Seite bzw. Folio aus der id <quelle>/<seite>_<system>; sonst jedes System für sich."""
    if os.environ.get("GRUPPE") != "seite":
        return i
    q, _, rest = i.partition("/")
    return q + "/" + rest.split("_")[0]


def bootstrap(a, b, ids, n=2000):
    g = defaultdict(list)
    for i in sorted(ids):
        g[gruppe(i)].append(i)
    g = list(g.values())
    rnd = random.Random(0)
    d = []
    for _ in range(n):
        z = [i for x in (rnd.choice(g) for _ in g) for i in x]
        nn = sum(a[i][1] for i in z) or 1
        d.append(100 * (sum(a[i][0] for i in z) - sum(b[i][0] for i in z)) / nn)
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]


def ser(f, ids):
    return 100 * sum(f[i][0] for i in ids) / max(sum(f[i][1] for i in ids), 1)


def main(ref_pfad, args):
    refz = [json.loads(z) for z in Path(ref_pfad).open(encoding="utf-8")]
    ref = {z["id"]: referenz(z) for z in refz}
    alias = {z["alias"]: z["id"] for z in refz if "alias" in z}
    quelle = {z["id"]: z["quelle"] for z in refz}
    gruppen = defaultdict(set)
    for i, q in quelle.items():
        gruppen[q].add(i)
        gruppen["alle"].add(i)
    erg = {n: fehler(ref, h) for n, h in (lesungen(a, set(ref), alias) for a in args)}
    print(
        f"Referenz {ref_pfad}: {len(ref)} Systeme, {sum(len(r[0]) for r in ref.values())} Töne"
    )
    print("| Quelle | " + " | ".join(erg) + " |")
    print("|---" * (len(erg) + 1) + "|")
    for q in sorted(gruppen, key=lambda q: (q == "alle", q)):
        print(
            f"| {q} ({len(gruppen[q])}) | "
            + " | ".join(f"{ser(f, gruppen[q]):.2f}" for f in erg.values())
            + " |"
        )
    zeile = []
    for f in erg.values():
        s = [f[i][2] for i in ref if f[i][2] is not None]
        zeile.append(f"{100 * sum(s) / max(len(s), 1):.1f}")
    print("| Anfangsschlüssel richtig (%) | " + " | ".join(zeile) + " |")
    namen = list(erg)
    for x in range(len(namen)):
        for y in range(x + 1, len(namen)):
            a, b = erg[namen[x]], erg[namen[y]]
            d = ser(a, ref) - ser(b, ref)
            lo, hi = bootstrap(a, b, ref)
            print(
                f"{namen[x]} − {namen[y]}: {d:+.2f} Punkte, 95 % [{lo:+.2f}, {hi:+.2f}]"
            )


def selbsttest():
    t, s = folge("c4 z2 l3 , z2 l2 L z1 l2")
    assert t == list("ghgfef") and s == "c4", (t, s)
    t2, s2 = folge("(<m>c<m>4) de(<m>g<m>h) us(<m>g<m>V<m>f) (<m>:) me(<m>e<m>f)")
    assert t2 == t and s2 == "c4", (t2, s2)
    ref = {"a": (list("abc"), ["c4"]), "b": (list("de"), None)}
    f = fehler(ref, {"a": (list("abd"), "c4"), "b": (list("de"), "f3")})
    assert f == {"a": (1, 3, True), "b": (0, 2, None)}, f
    assert abs(ser(f, ref) - 20.0) < 1e-9
    lo, hi = bootstrap(f, f, ref)
    assert lo == hi == 0
    print("Selbsttest in Ordnung")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        selbsttest()
    else:
        main(sys.argv[1], sys.argv[2:])
