"""Auswertung des Prüfbogens „Schlüssel am Bild“ gegen die lokale Zuordnung (lehren_schluessel.md).

    python3 -I skripte/pruefbogen_auswerten.py <Ordner mit urteile/*.json> [daten/pruefbogen_produktion]

Je Urteil: Welche Herkunft (referenz, ohne = Lesung ohne Vorwissen, mit = Lesung mit Vorwissen) hat die gewählte
Option? Die Option „kein Schlüssel am Zeilenanfang“ bedeutet laut Projektleiter (19.11.2026): Zeilenanfang im Bild
abgeschnitten, nicht beurteilbar; sie zählt wie „unsicher“ als nicht beurteilt. „anderer“: keiner der Vorschläge.
Ausgabe je Menge: Zahl der beurteilten Zeilen, wie oft der Schlüssel der Referenz, der Lesung ohne und der Lesung mit
Vorwissen am Bild richtig ist, und wie oft das Vorwissen einen richtigen Schlüssel verdirbt bzw. einen falschen
richtigstellt.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from auswertung_vorwissen import schluessel  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
BOGEN = HERE / (
    sys.argv[2] if len(sys.argv) > 2 else "daten/pruefbogen_schluessel"
)  # anderer Bogen: 2. Argument


def main(ordner):
    zu = json.loads((BOGEN / "zuordnung.json").read_text())
    daten = {z["id"]: z for z in json.loads((BOGEN / "daten.json").read_text())}
    je = defaultdict(Counter)
    for f in sorted(Path(ordner).glob("urteile/*.json")):
        u = json.loads(f.read_text())
        pid = f.stem
        x = je[zu[pid]["menge"]]
        x["urteile"] += 1
        wahl = u.get("wahl")
        text = {o["o"]: o["text"] for o in daten[pid]["optionen"]}.get(wahl, "")
        notiz = (u.get("notiz") or "").strip()
        if wahl == "anderer" and "kein schlüssel" in notiz.lower():
            text = "kein Schlüssel am Zeilenanfang"
        if wahl in (None, "unsicher") or text == "kein Schlüssel am Zeilenanfang":
            x["nicht beurteilbar"] += 1
            continue
        x["beurteilt"] += 1
        if wahl == "anderer":
            # Notiz mit Schlüsselbuchstaben auf den Linien 1-4 (z. B. "D F A C"): mit den Vorschlägen vergleichen
            richtig = set()
            m = re.match(r"nur ([CFcf]) auf Linie (\d)", notiz)
            if m:  # „nur F auf Linie 2, kein C auf Linie 4“: Schreibweise; für die Tonhöhe zählt der genannte Schlüssel
                eff = schluessel([m[1].lower() + m[2]])
                richtig = {
                    h
                    for h, t in zu[pid]["kandidaten"].items()
                    if schluessel([t]) == eff
                }
                x["Schlüssel aus Notiz"] += 1
            elif re.fullmatch(r"([A-Ga-g]\s*){2,5}", notiz):
                eff = schluessel(
                    [
                        "/".join(
                            f"{b.lower()}{k}" for k, b in enumerate(notiz.split(), 1)
                        )
                    ]
                )
                richtig = {
                    h
                    for h, t in zu[pid]["kandidaten"].items()
                    if schluessel([t]) == eff
                }
                x["Buchstaben aus Notiz"] += 1
            if not richtig:
                x["keiner der Vorschläge"] += 1
        else:
            richtig = set(zu[pid]["optionen"][wahl])
        for h in ("referenz", "ohne", "mit"):
            x[f"{h} richtig"] += h in richtig
        x["Vorwissen verdirbt"] += "ohne" in richtig and "mit" not in richtig
        x["Vorwissen stellt richtig"] += "mit" in richtig and "ohne" not in richtig
        x["Referenz falsch"] += "referenz" not in richtig
    felder = [
        "urteile",
        "nicht beurteilbar",
        "beurteilt",
        "referenz richtig",
        "ohne richtig",
        "mit richtig",
        "Vorwissen stellt richtig",
        "Vorwissen verdirbt",
        "Referenz falsch",
        "keiner der Vorschläge",
        "Buchstaben aus Notiz",
        "Schlüssel aus Notiz",
    ]
    print("| Menge | " + " | ".join(felder) + " |")
    print("|---" * (len(felder) + 1) + "|")
    for m, x in je.items():
        print(f"| {m} | " + " | ".join(str(x[k]) for k in felder) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
