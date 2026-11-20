"""Melodie-RAG, Schlüsselwechsel transpositionsbewusst (Punkt 1). Grundlage: Kandidaten aus rag_kandidaten2.py.
Referenz je Zeile = beste Lesung ohne Vorwissen (CTC + Attention). Ein Kandidat mit anderem Schlüssel verschiebt alle
Tonhöhen um d Stufen gegenüber der Referenz. Mehrheitslage m = häufigste Verschiebung, bei der die fremden Fassungen am
besten zur Referenz passen (Kandidaten-Verschiebungen −7…7). Varianten:
  A  eigenes Gewicht rho2 für den Konkordanzwert von Kandidaten mit Schlüsselwechsel (rho für die übrigen)
  B  Konkordanzwert eines Schlüsselwechsels nur, wenn d = m (Mehrheitslage), sonst 0
  AB beides
Gewichte auf der Validierung nach Melodie-SER gewählt (das Maß, das für das Nachschlagen zählt).
Aufruf: ../natural_music_processing/.venv_helsen/bin/python -I skripte/rag_bewertung2.py fest2 loso1012_2"""

import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_bewertung as R  # noqa: E402

HERE = R.HERE
G = R.GAMUT
LAM, BETAS, MUS, RHOS, RHO2S = (
    0.3,
    [0.0, 0.5],
    [0.0, 0.05, 0.1],
    [0.0, 0.05, 0.1, 0.2, 0.5],
    [0.0, 0.02, 0.05, 0.1, 0.2, 0.5],
)


def shift(p, d):
    return "".join(G[G.index(c) + d] for c in p if 0 <= G.index(c) + d < len(G))


def prepare(tag, split, konk):
    rows = R.load(tag, split, konk)
    for r in rows:
        q = r["key"].split("/")[0]
        own = konk["quelle_von"].get(q)
        vers = sorted(
            {
                R.collapse(t)
                for c in konk["zeilen"].get(r["key"], [])
                for s, t in konk["fassungen"].get(c, [])
                if s != own
            }
        )
        sc0 = [
            LAM * r["ctc"][i] + (1 - LAM) * r["att"][i] for i in range(len(r["cands"]))
        ]
        ref = max(range(len(sc0)), key=sc0.__getitem__)
        pit = [R.pitches(c) or "" for c in r["cands"]]
        rp = pit[ref]
        d = [(G.index(p[0]) - G.index(rp[0])) if p and rp else 0 for p in pit]
        m = 0
        if rp and vers:
            votes = Counter()
            for v in vers:
                best = max(
                    range(-7, 8),
                    key=lambda s: (fuzz.partial_ratio(shift(rp, s), v), -abs(s)),
                )
                votes[best] += 1
            m = min(
                votes, key=lambda s: (-votes[s], abs(s), s)
            )  # Gleichstand: kleinste Verschiebung
        gp = R.pitches(r["gt"]) or ""
        r.update(d=d, m=m, med=[Levenshtein.distance(p, gp) for p in pit], mlen=len(gp))
    return rows


def picks(rows, beta, mu, rho, rho2, maj):
    out = []
    for r in rows:
        best, bs = 0, -1e18
        for i in range(len(r["cands"])):
            wechsel = r["d"][i] != 0
            rag = r["rag"][i]
            if wechsel and maj and r["d"][i] != r["m"]:
                rag = 0.0
            s = (
                LAM * r["ctc"][i]
                + (1 - LAM) * r["att"][i]
                + mu * r["lm"][i]
                + beta * len(r["cands"][i])
                + (rho2 if wechsel else rho) * rag
            )
            if s > bs:
                best, bs = i, s
        out.append(best)
    return out


def metrics(rows, p):
    tok = (
        100 * sum(r["ed"][i] for r, i in zip(rows, p)) / sum(len(r["gt"]) for r in rows)
    )
    mel = (
        100
        * sum(r["med"][i] for r, i in zip(rows, p) if r["mlen"])
        / sum(r["mlen"] for r in rows if r["mlen"])
    )
    return tok, mel


def main(tags):
    konk = json.load((HERE / "daten" / "konkordanz" / "konkordanzen.json").open())
    lines = [
        "## Schlüsselwechsel transpositionsbewusst (`skripte/rag_bewertung2.py`)",
        "",
        "Gewichte (beta, mu, rho, rho2) auf der Validierung nach Melodie-SER gewählt; lam = 0,3. A = eigenes Gewicht für Schlüsselwechsel, B = Schlüsselwechsel nur in der Mehrheitslage der Fassungen.",
        "",
        "| Testmenge | Variante | Gewichte | Val Melodie-SER | Test Token-SER | Test Melodie-SER |",
        "|---|---|---|---|---|---|",
    ]
    for tag in tags:
        va, te = prepare(tag, "val", konk), prepare(tag, "test", konk)
        varianten = {
            "ohne Vorwissen": [(b, 0.0, 0.0, 0.0, False) for b in BETAS],
            "bisher (ein rho für alle)": [
                (b, mu, rho, rho, False) for b, mu, rho in product(BETAS, MUS, RHOS)
            ],
            "A": [
                (b, mu, rho, r2, False)
                for b, mu, rho, r2 in product(BETAS, MUS, RHOS, RHO2S)
            ],
            "B": [(b, mu, rho, rho, True) for b, mu, rho in product(BETAS, MUS, RHOS)],
            "AB": [
                (b, mu, rho, r2, True)
                for b, mu, rho, r2 in product(BETAS, MUS, RHOS, RHO2S)
            ],
        }
        for name, grid in varianten.items():
            k = min(grid, key=lambda g: metrics(va, picks(va, *g))[1])
            vm = metrics(va, picks(va, *k))[1]
            tt, tm = metrics(te, picks(te, *k))
            lines.append(
                f"| {tag} | {name} | {k[:4]} | {vm:.2f} | {tt:.2f} | {tm:.2f} |"
            )
            print(lines[-1], flush=True)
        mtr = Counter(r["m"] for r in te)
        lines.append(
            f"| {tag} | Mehrheitslage m der Testzeilen | {dict(mtr.most_common(5))} | | | |"
        )
    with (HERE / "docs" / "rag.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


if __name__ == "__main__":
    assert shift("dfh", 4) == "hkm"
    main(sys.argv[1:])
