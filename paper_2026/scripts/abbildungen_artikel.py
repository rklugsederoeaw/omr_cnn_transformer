"""Abbildungen des CRNN-Artikels (docs/arxiv/neufassung/paper.md).

    /home/rklug/projects_2026/omr_cli/.venv/bin/python -I skripte/abbildungen_artikel.py

Abb. 1: ein Testsystem der festen Testmenge (CH-E 611, 223v_11) mit seiner Zielfolge, Bild unverändert.
Abb. 2: Blockschaltbild des Modells und der Lesekette (Zahlen aus Abschnitt 4 des Artikels).
Ausgabe: docs/arxiv/neufassung/abb/abb1_kodierung.{pdf,png}, abb2_modell.{pdf,png}.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
AUS = HERE / "docs/arxiv/neufassung/abb"
OMR = HERE.parent / "trainingsdaten/omr_gt"


def abb1():
    x = next(
        z
        for z in map(json.loads, (OMR / "music/index.jsonl").open())
        if z["source"] == "sbe-0611" and z["id"] == "223v_11" and z["split"] == "test"
    )
    bild = Image.open(OMR / x["image"]).convert("L")
    w, h = bild.size
    hb = 7.2 * h / w  # Bildhöhe in Zoll bei 7,2 Zoll Breite
    fig = plt.figure(figsize=(7.2, hb + 0.35))
    ax = fig.add_axes([0, 0.35 / (hb + 0.35), 1, hb / (hb + 0.35)])
    ax.imshow(bild, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    fig.text(0.005, 0.12 / (hb + 0.35), x["label"], family="monospace", fontsize=7.4)
    for s in ("pdf", "png"):
        fig.savefig(AUS / f"abb1_kodierung.{s}", dpi=300)
    plt.close(fig)
    return x["label"]


def kasten(ax, x, y, b, h, text, farbe="#eef1f6"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            b,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            fc=farbe,
            ec="#3b4252",
            lw=0.8,
        )
    )
    ax.text(x + b / 2, y + h / 2, text, ha="center", va="center", fontsize=7)


def pfeil(ax, x0, y0, x1, y1, text=""):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#3b4252"),
    )
    if text:
        ax.text(
            (x0 + x1) / 2 + 0.05,
            (y0 + y1) / 2,
            text,
            fontsize=6.3,
            color="#3b4252",
            va="center",
        )


def abb2():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # obere Reihe: Modell
    kasten(ax, 0.1, 3.9, 1.8, 0.8, "staff image\ngrey, height 128")
    kasten(ax, 2.3, 3.9, 2.0, 0.8, "CNN\n4 conv. blocks\n32–256 channels")
    kasten(ax, 4.7, 3.9, 2.0, 0.8, "BiLSTM\n2 layers, 2 × 256")
    kasten(ax, 7.4, 4.3, 2.4, 0.6, "CTC head\n59 tokens + blank")
    kasten(ax, 7.4, 3.25, 2.4, 0.75, "Transformer decoder\n6 layers, d 256, 4 heads")
    kasten(
        ax, 4.7, 2.45, 2.0, 0.75, "melody model\n(Cantus Database\nmelodies)", "#f4ece2"
    )
    pfeil(ax, 1.9, 4.3, 2.3, 4.3)
    pfeil(ax, 4.3, 4.3, 4.7, 4.3)
    pfeil(ax, 6.7, 4.45, 7.4, 4.6)
    pfeil(ax, 6.7, 4.1, 7.4, 3.75)
    pfeil(ax, 6.7, 2.95, 7.4, 3.4)
    ax.text(
        6.95,
        2.62,
        "initialises\nself-attention\nand feed-forward",
        fontsize=6.0,
        color="#3b4252",
        va="center",
    )
    ax.text(
        0.1,
        3.45,
        "training: 0.3 · CTC + 0.7 · cross-entropy",
        fontsize=6.3,
        color="#3b4252",
    )
    # untere Reihe: Lesen
    kasten(
        ax,
        0.1,
        0.3,
        3.0,
        1.2,
        "concordance score C\nversions of the same chant\nin other manuscripts\n(Cantus Database inventory)",
        "#f4ece2",
    )
    kasten(
        ax,
        3.5,
        0.3,
        3.0,
        1.2,
        "score of each candidate\nλ log P_CTC + (1−λ) log P_att\n+ β |y| + μ log P_LM + ρ C(y)",
    )
    kasten(
        ax,
        6.9,
        0.3,
        2.9,
        1.2,
        "candidates\nup to 12 beam hypotheses\n+ CTC greedy reading\n+ clef variants",
    )
    pfeil(ax, 8.6, 3.25, 8.35, 1.5)
    pfeil(ax, 9.82, 4.6, 9.75, 1.5)  # CTC-Pfad: Greedy-Lesung und CTC-Bewertung
    ax.text(9.84, 2.6, "CTC", fontsize=6.0, color="#3b4252")
    pfeil(ax, 6.9, 0.9, 6.5, 0.9)
    pfeil(ax, 3.1, 0.9, 3.5, 0.9)
    pfeil(ax, 5.5, 2.45, 5.2, 1.5)
    ax.text(5.42, 2.0, "μ log P_LM", fontsize=6.0, color="#3b4252")
    for s in ("pdf", "png"):
        fig.savefig(AUS / f"abb2_modell.{s}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    AUS.mkdir(parents=True, exist_ok=True)
    print(abb1())
    abb2()
    print("Abbildungen in", AUS)
