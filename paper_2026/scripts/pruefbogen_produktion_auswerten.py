"""Auswertung des Produktionsbogens (Melodiemodell ändert den Anfangsschlüssel), Artikel Abschnitt 6.2.

    python3 -I skripte/pruefbogen_produktion_auswerten.py daten/pruefbogen_produktion/urteile_2

Eigene Antwortsemantik dieses Bogens (anders als pruefbogen_auswerten.py): Die Option „kein Schlüssel am Zeilenanfang“
ist die Lesung ohne Melodiemodell, die keinen Schlüssel hat. Die Notiz des Prüfers trennt „kein schlüssel“ (am Bild
steht wirklich kein Schlüssel; die Lesung ohne Melodiemodell ist richtig) von „abgeschnitten“ (Anfang nicht sichtbar,
nicht beurteilbar). „anderer“ = keine der beiden Lesungen richtig, „unsicher“ = nicht beurteilbar.
"""

import json
import sys
from collections import Counter
from pathlib import Path

BOGEN = Path(__file__).resolve().parent.parent / "daten/pruefbogen_produktion"


def klasse(wahl, text, herkunft, notiz):
    if wahl in ("anderer", "unsicher"):
        return wahl
    if text.startswith("kein Schlüssel"):
        return (
            "abgeschnitten"
            if "abgeschnitten" in notiz.lower()
            else "ohne: kein Schlüssel geschrieben"
        )
    return f"{herkunft}: Schlüssel sichtbar"


def main(urteile):
    zu = json.load((BOGEN / "zuordnung.json").open())
    opt = {
        z["id"]: {o["o"]: o["text"] for o in z["optionen"]}
        for z in json.load((BOGEN / "daten.json").open())
    }
    c = Counter()
    for p in sorted(zu):
        u = json.load((Path(urteile) / "urteile" / f"{p}.json").open())
        w = u.get("wahl")
        c[
            klasse(
                w,
                opt[p].get(w, ""),
                zu[p]["optionen"].get(w, [w])[0],
                u.get("notiz") or "",
            )
        ] += 1
    for k, v in sorted(c.items()):
        print(f"{v:3d}  {k}")
    print(f"{sum(c.values()):3d}  zusammen")


if __name__ == "__main__":
    assert (
        klasse("o0", "kein Schlüssel am Zeilenanfang", "ohne", "Abgeschnitten")
        == "abgeschnitten"
    )
    assert (
        klasse("o1", "kein Schlüssel am Zeilenanfang", "ohne", "kein schlüssel")
        == "ohne: kein Schlüssel geschrieben"
    )
    assert klasse("o1", "C auf Linie 4", "mit", "") == "mit: Schlüssel sichtbar"
    main(sys.argv[1])
