"""Gesamtdatensatz wie general_v7, aber ohne die fünf Prüfhandschriften von LightOn (Messreihe M4, fairer Vergleich).

    python3 -I skripte/general_ohne_pruef.py     # schreibt daten/general_ohne_pruef/music/index.jsonl

general_v7 = alte Volpiano-GT (volpiano4) und 27 geprüfte Transkribus-Quellen, darunter viele gotische Chorbücher.
LightOn seiten_d hat die fünf Prüfhandschriften (A-Wn 15484, A-Geras 6, A-KN 65, A-Gu 17, A-VOR 254) nie gesehen;
hier fallen sie aus allen drei Teilen. Bildpfade relativ zu daten/general_ohne_pruef (Cluster-Abbild).
"""

import json
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PRUEF = {"a-wn-15484", "a-geras-6", "a-kn-65", "a-gu-17", "a-vor-254"}
ALT = HERE / "daten/general_v7"
NEU = HERE / "daten/general_ohne_pruef"


def main():
    zeilen = []
    for x in map(json.loads, (ALT / "music/index.jsonl").open()):
        if x["source"] in PRUEF:
            continue
        bild = Path(x["image"])
        bild = bild if bild.is_absolute() else ALT / bild
        zeilen.append(dict(x, image=os.path.relpath(bild, NEU)))
    (NEU / "music").mkdir(parents=True, exist_ok=True)
    (NEU / "music/index.jsonl").write_text(
        "".join(json.dumps(z, ensure_ascii=False) + "\n" for z in zeilen)
    )
    assert all((NEU / z["image"]).is_file() for z in zeilen)
    assert not PRUEF & {z["source"] for z in zeilen}
    print(Counter(z["split"] for z in zeilen))


if __name__ == "__main__":
    main()
