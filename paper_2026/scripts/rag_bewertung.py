"""Melodie-RAG, CPU-Teil: Kandidaten (rag_kandidaten.py) zusätzlich nach ihrer Nähe zu den Konkordanzen bewerten.
Konkordanzwert eines Kandidaten = beste Teilstring-Ähnlichkeit (rapidfuzz partial_ratio) seiner Tonfolge mit einer
fremden Fassung eines Gesangs seines Folios (Tonhöhen über den eigenen Schlüssel des Kandidaten, Wiederholungen
zusammengezogen). Gesamtbewertung: lam·CTC + (1−lam)·Att + mu·LM + beta·Länge + rho·Noten·Konkordanz. Gewichte auf
der Validierung gewählt. Aufruf: ../natural_music_processing/.venv_helsen/bin/python -I skripte/rag_bewertung.py <TAG>"""

import json
import re
import sys
from itertools import product
from pathlib import Path

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "scripts"))
from volpiano_geo import CLEF_NOTE, GAMUT, pos_num  # noqa: E402

POS = re.compile(r"^[lz]-?\d+$")
LAMS, BETAS, MUS, RHOS = (
    [0.2, 0.3, 0.5],
    [0.0, 0.5, 1.0],
    [0.0, 0.05, 0.1],
    [0.0, 0.02, 0.05, 0.1, 0.2, 0.5],
)


def collapse(s):
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])


def pitches(tokens):
    """Geometrische Tokens -> Volpiano-Tonhöhen über den jeweils gültigen Schlüssel; None ohne Anfangsschlüssel."""
    out, clef = [], None
    for t in tokens:
        if re.fullmatch(r"[a-g]-?\d(/[a-g]\d)*", t):
            eff = next((p for p in t.split("/") if p[0] in "cf"), None)
            if eff:
                clef = (CLEF_NOTE[eff[0]], 2 * (int(eff[1:]) - 1))
        elif POS.match(t) and clef:
            i = pos_num(t) - clef[1] + clef[0]
            if 0 <= i < len(GAMUT):
                out.append(GAMUT[i])
    return collapse("".join(out)) if clef else None


def lev(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


def load(tag, split, konk):
    rows = []
    for line in (
        HERE / "daten" / "konkordanz" / f"kandidaten_{tag}_{split}.jsonl"
    ).open():
        r = json.loads(line)
        quelle = r["key"].split("/")[0]
        own = konk["quelle_von"].get(quelle)
        vers = sorted(
            {
                collapse(t)
                for c in konk["zeilen"].get(r["key"], [])
                for s, t in konk["fassungen"].get(c, [])
                if s != own
            }
        )
        rag = []
        for cand in r["cands"]:
            p = pitches(cand)
            if not p or not vers:
                rag.append(0.0)
                continue
            best = process.extractOne(p, vers, scorer=fuzz.partial_ratio)
            rag.append(best[1] / 100 * sum(1 for t in cand if POS.match(t)))
        r["rag"] = rag
        r["ed"] = [Levenshtein.distance(c, r["gt"]) for c in r["cands"]]
        r["hat_konk"] = bool(vers)
        rows.append(r)
    return rows


def ser(rows, lam, beta, mu, rho, pick_only=False):
    ed = L = 0
    picks = []
    for r in rows:
        sc = [
            lam * r["ctc"][i]
            + (1 - lam) * r["att"][i]
            + mu * r["lm"][i]
            + beta * len(r["cands"][i])
            + rho * r["rag"][i]
            for i in range(len(r["cands"]))
        ]
        b = max(range(len(sc)), key=sc.__getitem__)
        picks.append(b)
        ed += r["ed"][b]
        L += len(r["gt"])
    return picks if pick_only else 100 * ed / L


def main(tag):
    konk = json.load((HERE / "daten" / "konkordanz" / "konkordanzen.json").open())
    va, te = load(tag, "val", konk), load(tag, "test", konk)
    grid = {k: ser(va, *k) for k in product(LAMS, BETAS, MUS, RHOS)}
    best = lambda cond: min((k for k in grid if cond(k)), key=grid.get)  # noqa: E731
    cfg = {
        "gemeinsam (CTC + Attention)": best(lambda k: k[2] == 0 and k[3] == 0),
        "+ Melodiemodell": best(lambda k: k[3] == 0),
        "+ Konkordanzen (RAG)": best(lambda k: k[2] == 0),
        "+ Melodiemodell + Konkordanzen": best(lambda k: True),
    }

    def orakel(rows):
        return 100 * sum(min(r["ed"]) for r in rows) / sum(len(r["gt"]) for r in rows)

    lines = [
        f"## Melodie-RAG: {tag}",
        "",
        f"`skripte/rag_bewertung.py`; {len(va)} Validierungs-, {len(te)} Testzeilen; Testzeilen mit fremden Konkordanzen: {sum(r['hat_konk'] for r in te)}. Gewichte (lam, beta, mu, rho) je Variante auf der Validierung gewählt.",
        "",
        "| Variante | Gewichte | Val-SER | Test-SER |",
        "|---|---|---|---|",
    ]
    for name, k in cfg.items():
        lines.append(f"| {name} | {k} | {grid[k]:.2f} | {ser(te, *k):.2f} |")
    lines.append(
        f"| Orakel: bester Kandidat je Zeile | | {orakel(va):.2f} | {orakel(te):.2f} |"
    )
    k0, k1 = cfg["gemeinsam (CTC + Attention)"], cfg["+ Konkordanzen (RAG)"]
    p0, p1 = ser(te, *k0, pick_only=True), ser(te, *k1, pick_only=True)
    ch = [(r["ed"][b] - r["ed"][a]) for r, a, b in zip(te, p0, p1) if a != b]
    lines += [
        "",
        f"Testzeilen, deren Lesung die Konkordanzen ändern: {len(ch)}; besser {sum(d < 0 for d in ch)}, schlechter {sum(d > 0 for d in ch)}, gleich {sum(d == 0 for d in ch)}.",
    ]
    for name, sel in [
        ("mit Konkordanzen", lambda r: r["hat_konk"]),
        ("ohne Konkordanzen", lambda r: not r["hat_konk"]),
    ]:
        sub = [r for r in te if sel(r)]
        if sub:
            lines.append(
                f"Test-SER {name} ({len(sub)} Zeilen): ohne RAG {ser(sub, *k0):.2f}, mit RAG {ser(sub, *k1):.2f}."
            )
    (HERE / "daten" / "konkordanz" / f"gewichte_{tag}.json").write_text(
        json.dumps(cfg, ensure_ascii=False)
    )
    out = HERE / "docs" / "rag.md"
    with out.open("a") as fh:
        fh.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


if __name__ == "__main__":
    assert pitches(["c4", "l4", "z3", ",", "l3"]) == "kjh"
    assert pitches(["l3"]) is None
    main(sys.argv[1])
