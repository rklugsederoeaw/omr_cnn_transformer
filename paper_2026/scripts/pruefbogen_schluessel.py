"""Prüfbogen „Schlüssel am Bild“ (Gutachten K1): Zeilen, in denen das Vorwissen den Anfangsschlüssel ändert.

    python3 -I skripte/pruefbogen_schluessel.py

Je Zeile bis zu drei wirksam verschiedene Anfangsschlüssel: Referenz (aus der Cantus Database übertragen), Lesung
ohne Vorwissen, Lesung mit Melodiemodell und Konkordanzen (lesen_reihe.py, Stufe voll). Der Bogen zeigt sie blind
(ohne Herkunft, gemischt); die Zuordnung bleibt in zuordnung.json und wird nicht veröffentlicht.
Auswahl: feste Testmenge (M1 Seed 0) und A-KN 1012 vollständig, F-Pn lat. 12044 und A-Wn 1799 je 30 (Zufall, Startwert 0).
Ausgabe: daten/pruefbogen_schluessel/{daten.json, zuordnung.json, bilder/}.
"""

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auswertung_vorwissen import SCHL, key_of, schluessel  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
AUS = HERE / "daten/pruefbogen_schluessel"
MENGEN = [  # (Name, Index, Lesungen, Höchstzahl)
    (
        "feste Testmenge",
        HERE.parent / "trainingsdaten/omr_gt/music/index.jsonl",
        "m1_hybrid_s0",
        None,
    ),
    (
        "A-KN 1012",
        HERE / "daten/loso_a-kn-1012/music/index.jsonl",
        "m2_a-kn-1012",
        None,
    ),
    (
        "F-Pn lat. 12044",
        HERE / "daten/loso_f-pn-lat-12044/music/index.jsonl",
        "m2_f-pn-lat-12044",
        30,
    ),
    ("A-Wn 1799", HERE / "daten/loso_a-wn-1799/music/index.jsonl", "m2_a-wn-1799", 30),
]


def linien(t):
    """Buchstaben auf den Linien 1 bis 4, wie man sie im Bild liest: c4 und f2/a3/c4/e5 ergeben beide 'F A C E' bzw.
    'D F A C'. Grundlage ist der wirksame Schlüssel (erster C- oder F-Teil)."""
    eff = next((x for x in t.split("/") if x[0] in "cf"), None)
    if not eff:
        return ""
    b, n = "abcdefg".index(eff[0]), int(eff[1:])
    return " ".join("abcdefg"[(b + 2 * (k - n)) % 7].upper() for k in range(1, 5))


def klartext(t):
    """c4 -> 'C auf Linie 4'; zusammengesetzte Schlüssel Teil für Teil; Note statt Schlüssel -> kein Schlüssel."""
    if not SCHL.match(t):
        return "kein Schlüssel am Zeilenanfang"
    return " + ".join(f"{x[0].upper()} auf Linie {x[1:]}" for x in t.split("/"))


def main():
    shutil.rmtree(
        AUS / "bilder", ignore_errors=True
    )  # nur die Bilder: index.html und Urteile bleiben
    (AUS / "bilder").mkdir(parents=True)
    daten, zuordnung = [], {}
    rnd = random.Random(0)
    for name, index, les, hoechst in MENGEN:
        ref = {}
        for z in map(json.loads, index.open()):
            if z["split"] == "test":
                bild = Path(z["image"])
                ref.setdefault(
                    key_of(z["image"]),
                    (z, bild if bild.is_absolute() else index.parent.parent / bild),
                )
        zeilen = []
        for x in map(
            json.loads, (HERE / "daten/lesen_rag" / les / "lesungen_voll.jsonl").open()
        ):
            if x["key"] in ref and schluessel(x["gelesen"]) != schluessel(
                x["ohne_vorwissen"]
            ):
                zeilen.append(x)
        if hoechst:
            zeilen = rnd.sample(zeilen, min(hoechst, len(zeilen)))
        for x in zeilen:
            z, bild = ref[x["key"]]
            kand = {
                "referenz": z["label"].split()[0],
                "ohne": x["ohne_vorwissen"][0],
                "mit": x["gelesen"][0],
            }
            optionen = {}  # wirksamer Schlüssel -> {Tokens, Herkunft}
            for herkunft, tok in kand.items():
                e = schluessel([tok])
                o = optionen.setdefault(e, {"tokens": [], "herkunft": []})
                if tok not in o["tokens"]:
                    o["tokens"].append(tok)
                o["herkunft"].append(herkunft)
            liste = list(optionen.values())
            rnd.shuffle(liste)
            pid = f"z{len(daten):03d}"
            ziel = AUS / "bilder" / f"{pid}.jpg"
            shutil.copy(bild, ziel)
            daten.append(
                {
                    "id": pid,
                    "menge": name,
                    "bild": f"bilder/{pid}.jpg",
                    "optionen": [
                        {
                            "o": f"o{k}",
                            "text": " = ".join(klartext(t) for t in o["tokens"])
                            + (
                                f" · Linien 1–4: {linien(o['tokens'][0])}"
                                if SCHL.match(o["tokens"][0])
                                else ""
                            ),
                            "tokens": o["tokens"],
                        }
                        for k, o in enumerate(liste)
                    ],
                }
            )
            zuordnung[pid] = {
                "menge": name,
                "key": x["key"],
                "id": z["id"],
                "kandidaten": kand,
                "optionen": {f"o{k}": o["herkunft"] for k, o in enumerate(liste)},
            }
        print(name, len(zeilen))
    (AUS / "daten.json").write_text(json.dumps(daten, ensure_ascii=False))
    (AUS / "zuordnung.json").write_text(
        json.dumps(zuordnung, ensure_ascii=False, indent=1)
    )
    groesse = sum(p.stat().st_size for p in (AUS / "bilder").iterdir())
    print(len(daten), "Zeilen,", round(groesse / 1e6, 1), "MB Bilder")


if __name__ == "__main__":
    main()
