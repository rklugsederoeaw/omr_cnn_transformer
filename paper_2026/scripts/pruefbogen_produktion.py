"""Prüfbogen „Schlüssel am Bild“, Produktion: Stichprobe der Lesungen, in denen das Melodiemodell den Anfangsschlüssel
geändert hat (transkription/<quelle>.tsv, herkunft lesung_v6, schluessel_geaendert; lehren_schluessel.md, Variante A).

    .venv_helsen/bin/python -I skripte/pruefbogen_produktion.py

60 Zeilen aus allen markierten (Zufall, Startwert 0). Je Zeile zwei Vorschläge, blind und gemischt: Anfangsschlüssel mit
Melodiemodell (tokens, so exportiert) und ohne (ohne_vorwissen). Bild: Zuschnitt aus dem Leseindex
daten/lesen_rag/<quelle>_neu_v6/music/index.jsonl (Arbeitsindex der Lesung). Ausgabe: daten/pruefbogen_produktion/{daten.json, zuordnung.json, bilder/};
die Seite (index.html) liegt daneben und wird hier nicht angefasst.
"""

import csv
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auswertung_vorwissen import SCHL, schluessel  # noqa: E402
from pruefbogen_schluessel import klartext, linien  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
AUS = HERE / "daten/pruefbogen_produktion"
N = 60


def main():
    kandidaten = []
    for tsv in sorted((HERE / "transkription").glob("*.tsv")):
        q = tsv.stem
        for r in csv.DictReader(tsv.open(), delimiter="\t"):
            if r["herkunft"] == "lesung_v6" and r["schluessel_geaendert"] == "True":
                if schluessel(r["tokens"].split()[:1]) != schluessel(
                    r["ohne_vorwissen"].split()[:1]
                ):
                    kandidaten.append(
                        (q, r)
                    )  # nur wirksam verschiedene Schlüssel (c4 = f2 zählt nicht)
    rnd = random.Random(0)
    auswahl = rnd.sample(kandidaten, N)
    shutil.rmtree(AUS / "bilder", ignore_errors=True)
    (AUS / "bilder").mkdir(parents=True)
    indizes, daten, zuordnung = {}, [], {}
    for q, r in auswahl:
        if q not in indizes:
            indizes[q] = {
                x["id"]: x
                for x in map(
                    json.loads,
                    (HERE / "daten/lesen_rag" / f"{q}_neu_v6/music/index.jsonl").open(),
                )
            }
        zid = f"{int(r['bild']):04d}_{r['system']}"
        bild = Path(indizes[q][zid]["image"])
        kand = {"mit": r["tokens"].split()[0], "ohne": r["ohne_vorwissen"].split()[0]}
        optionen = [(h, t) for h, t in kand.items()]
        rnd.shuffle(optionen)
        pid = f"p{len(daten):03d}"
        shutil.copy(bild, AUS / "bilder" / f"{pid}.jpg")
        daten.append(
            {
                "id": pid,
                "menge": q,
                "bild": f"bilder/{pid}.jpg",
                "optionen": [
                    {
                        "o": f"o{k}",
                        "text": klartext(t)
                        + (f" · Linien 1–4: {linien(t)}" if SCHL.match(t) else ""),
                        "tokens": [t],
                    }
                    for k, (_, t) in enumerate(optionen)
                ],
            }
        )
        zuordnung[pid] = {
            "menge": q,
            "id": zid,
            "kandidaten": kand,
            "optionen": {f"o{k}": [h] for k, (h, _) in enumerate(optionen)},
            "konfidenz": r["konfidenz"],
        }
    (AUS / "daten.json").write_text(json.dumps(daten, ensure_ascii=False))
    (AUS / "zuordnung.json").write_text(
        json.dumps(zuordnung, ensure_ascii=False, indent=1)
    )
    print(
        len(kandidaten),
        "markierte Lesungen,",
        N,
        "gezogen aus",
        len({q for q, _ in auswahl}),
        "Quellen",
    )


if __name__ == "__main__":
    main()
