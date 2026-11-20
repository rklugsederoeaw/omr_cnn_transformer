"""Gemeinsame Testmengen für den Vergleich CRNN gegen LightOnOCR (Messreihe M4, docs/arxiv/neufassung/messplan.md).

    python3 -I skripte/m4_mengen.py

bekannt: Systeme, die in der festen CRNN-Testmenge (trainingsdaten/omr_gt, split test) und im GABC-Testteil von
LightOn (LightOnOCR-3-1B/daten/korpus_gabc/test.jsonl) stehen, ohne CH-P 18 p. 55 (Mehrsystem-Leck, vorpruefung.md).
ungesehen: die 200 Prüfsysteme der fünf Prüfhandschriften (LightOnOCR-3-1B/daten/gt/systeme.jsonl); der
CRNN-Ausschnitt kommt aus dem Index der Quelle unter daten/<quelle>/music/index.jsonl (wie crnn_lesen.py).
Jedes Modell liest seinen eigenen Ausschnitt desselben Systems. Referenz ist das CRNN-Label (geprüfte GT).
Ausgabe je Menge: daten/vergleich_m4/<menge>.jsonl (id = <source>/<CRNN-id> wie der key von lesen_rag, alias = id bei
LightOn, quelle, label, lighton_bild) und ein CRNN-Index
daten/vergleich_m4/<menge>/music/index.jsonl (split test, Bildpfade relativ).
"""

import json
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
LIGHTON = HERE / "LightOnOCR-3-1B"
AUS = HERE / "daten/vergleich_m4"
GESPERRT = {("ch-p-18", "55")}  # (Quelle, Folio)


def jsonl(p):
    return [json.loads(z) for z in Path(p).open(encoding="utf-8")]


def schreiben(name, paare):
    """paare: [(Eintrag des CRNN-Index mit absolutem Bildpfad, LightOn-Eintrag)]"""
    basis = AUS / name
    (basis / "music").mkdir(parents=True, exist_ok=True)
    idx, liste = [], []
    for x, z in paare:
        idx.append(dict(x, image=os.path.relpath(x["image"], basis), split="test"))
        liste.append(
            {
                "id": f"{x['source']}/{x['id']}",  # wie der key von lesen_rag
                "alias": z["id"],  # id bei LightOn
                "quelle": z["quelle"],
                "label": x["label"],
                "lighton_bild": z["bild"],
            }
        )
    (basis / "music/index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in idx))
    (AUS / f"{name}.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in liste)
    )
    assert len({e["id"] for e in liste}) == len(liste)
    assert all((basis / e["image"]).is_file() for e in idx)
    print(name, len(liste), dict(Counter(e["quelle"] for e in liste)))


def bekannt():
    omr = ROOT / "trainingsdaten/omr_gt"
    test = {}
    for x in jsonl(omr / "music/index.jsonl"):
        if x["split"] == "test":
            test.setdefault(
                (x["source"], x["id"]),
                dict(x, image=str(omr / x["image"])),
            )
    paare, fehlt = [], Counter()
    for z in jsonl(LIGHTON / "daten/korpus_gabc/test.jsonl"):
        q, zweig, zid = z["id"].split("/")
        assert zweig == "export", z["id"]
        if (q, zid.split("_")[0]) in GESPERRT:
            fehlt["gesperrt"] += 1
        elif (q, zid) not in test:
            fehlt["nicht_in_fester_testmenge"] += 1
        else:
            paare.append((test[(q, zid)], z))
    print("bekannt: ausgelassen", dict(fehlt))
    schreiben("bekannt", paare)


def ungesehen():
    paare = []
    cache = {}
    for z in jsonl(LIGHTON / "daten/gt/systeme.jsonl"):
        q, zid = z["id"].split("/")
        if q not in cache:
            basis = HERE / "daten" / q
            cache[q] = {
                x["id"]: dict(x, image=str(basis / x["image"]))
                for x in jsonl(basis / "music/index.jsonl")
            }
        x = cache[q][zid]
        assert x["label"] == z["label"], z["id"]
        paare.append((x, z))
    schreiben("ungesehen", paare)


if __name__ == "__main__":
    bekannt()
    ungesehen()
