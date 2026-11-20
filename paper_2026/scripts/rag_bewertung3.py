"""Melodie-RAG, unsicherheitsgesteuert: Das Nachschlagen wirkt nur, wo das Bildmodell unsicher ist. Unsicherheit einer
Zeile = Abstand der besten zur zweitbesten Lesung unter CTC + Attention (Kandidaten mit gleicher Tokenfolge zählen
einmal); nur Zeilen mit Abstand < tau bekommen den Konkordanz- und Melodiemodellbeitrag. Optional Mehrheitslage (B).
Gewichte auf der Validierung nach Melodie-SER gewählt. Grundlage rag_kandidaten2.py; Vorbereitung aus rag_bewertung2.py.
Aufruf: ../natural_music_processing/.venv_helsen/bin/python -I skripte/rag_bewertung3.py loso1012_2 fest2"""

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag_bewertung2 as B  # noqa: E402

HERE = B.HERE
LAM = 0.3
BETAS, MUS, RHOS, TAUS = (
    [0.0, 0.5],
    [0.0, 0.05, 0.1],
    [0.0, 0.1, 0.2, 0.5],
    [0.5, 1.0, 2.0, 4.0, 8.0, 1e9],
)


def arrays(rows):
    out = []
    for r in rows:
        base = LAM * np.array(r["ctc"]) + (1 - LAM) * np.array(r["att"])
        s = np.sort(np.unique(np.round(base, 4)))[::-1]
        margin = s[0] - s[1] if len(s) > 1 else 1e9
        d = np.array(r["d"])
        out.append(
            dict(
                base=base,
                lm=np.array(r["lm"]),
                ln=np.array([len(c) for c in r["cands"]]),
                rag=np.array(r["rag"]),
                ragb=np.where((d != 0) & (d != r["m"]), 0.0, r["rag"]),
                ed=np.array(r["ed"]),
                med=np.array(r["med"]),
                glen=len(r["gt"]),
                mlen=r["mlen"],
                margin=margin,
            )
        )
    return out


def evaluate(A, beta, mu, rho, tau, maj):
    ed = L = med = Lm = 0
    for a in A:
        on = a["margin"] < tau
        s = (
            a["base"]
            + beta * a["ln"]
            + (mu * a["lm"] + rho * (a["ragb"] if maj else a["rag"]) if on else 0.0)
        )
        i = int(np.argmax(s))
        ed += a["ed"][i]
        L += a["glen"]
        if a["mlen"]:
            med += a["med"][i]
            Lm += a["mlen"]
    return 100 * ed / L, 100 * med / Lm


def main(tags):
    konk = json.load((HERE / "daten" / "konkordanz" / "konkordanzen.json").open())
    lines = [
        "## Unsicherheitsgesteuertes Nachschlagen (`skripte/rag_bewertung3.py`)",
        "",
        "Vorwissen (Melodiemodell mu, Konkordanzen rho) nur in Zeilen, deren beste und zweitbeste Lesung unter CTC + Attention weniger als tau auseinanderliegen; B = Schlüsselwechsel nur in der Mehrheitslage. Gewichte auf der Validierung (bekannte Quellen) nach Melodie-SER gewählt.",
        "",
        "| Testmenge | Variante | Gewichte (beta, mu, rho, tau, B) | Val Melodie-SER | Test Token-SER | Test Melodie-SER | Zeilen unsicher (Test) |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag in tags:
        va, te = (
            arrays(B.prepare(tag, "val", konk)),
            arrays(B.prepare(tag, "test", konk)),
        )
        for name, grid in {
            "ohne Vorwissen": [(b, 0.0, 0.0, 1e9, False) for b in BETAS],
            "immer (tau unendlich)": [
                (b, mu, rho, 1e9, maj)
                for b, mu, rho, maj in product(BETAS, MUS, RHOS, (False, True))
            ],
            "nur bei Unsicherheit": [
                (b, mu, rho, tau, maj)
                for b, mu, rho, tau, maj in product(
                    BETAS, MUS, RHOS, TAUS, (False, True)
                )
            ],
        }.items():
            k = min(grid, key=lambda g: evaluate(va, *g)[1])
            tt, tm = evaluate(te, *k)
            uns = sum(a["margin"] < k[3] for a in te) / len(te)
            lines.append(
                f"| {tag} | {name} | {k} | {evaluate(va, *k)[1]:.2f} | {tt:.2f} | {tm:.2f} | {uns:.0%} |"
            )
            print(lines[-1], flush=True)
        for tau in TAUS[:-1]:
            lines.append(
                f"| {tag} | Anteil unsicherer Zeilen bei tau = {tau} | | | Val {sum(a['margin'] < tau for a in va) / len(va):.0%} | Test {sum(a['margin'] < tau for a in te) / len(te):.0%} | |"
            )
    with (HERE / "docs" / "rag.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    main(sys.argv[1:])
