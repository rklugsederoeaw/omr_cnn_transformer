"""Melodie-RAG, Datenteil: Für jede Test- und Validierungszeile (feste Test- und Validierungsmenge des CRNN und alle Zeilen von A-KN 1012) die Gesänge
ihres Folios aus der CantusDB (dazu der letzte Gesang des Vorfolios, der oft überläuft) und deren Fassungen aus allen
anderen Quellen. Die eigene Quelle der Zeile und San Pietro B.79 (altrömisch) bleiben außen vor. Nur Tonhöhen (Volpiano-
Gamut, Kleinbuchstaben), Differenz nach 4 abgeschnitten. Datenbank nur lesend.
Ausgabe: daten/konkordanz/konkordanzen.json = {"zeilen": {"<quelle>/<id>": [cantus_id, ...]}, "fassungen": {cantus_id:
[[source_id, töne], ...]}, "quelle_von": {"<quelle>": source_id}}; Kennzahlen docs/konkordanz_daten.md.
Aufruf: python3 -I skripte/konkordanz_daten.py"""

import csv
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
G = "9abcdefghjklmnopqrs"
AUSSCHLUSS = {123661}


def folio_key(f):
    return [
        (0, int(x), "") if x.isdigit() else (1, 0, x)
        for x in re.findall(r"\d+|\D+", f or "")
    ]


def toene(v):
    return "".join(c for c in (v or "").split("4")[0].lower() if c in G)


def main():
    sid = {
        json.load(p.open())["slug"]: json.load(p.open())["source_id"]
        for p in (ROOT / "sources").glob("*.json")
    }
    db = sqlite3.connect(f"file:{ROOT / 'cantus_database.sqlite'}?mode=ro", uri=True)
    zeilen = [
        r
        for r in csv.DictReader(
            (ROOT / "trainingsdaten" / "zeilen.tsv").open(), delimiter="\t"
        )
        if r["split"] in ("test", "val") or r["quelle"] == "a-kn-1012"
    ]
    folios = defaultdict(list)  # source_id -> sortierte (folio, sequence, cantus_id)
    for s in {sid[r["quelle"]] for r in zeilen}:
        rows = db.execute(
            "select folio, sequence, cantus_id from chants where source_id=? and cantus_id is not null",
            (s,),
        ).fetchall()
        rows.sort(
            key=lambda x: (
                folio_key(x[0]),
                int(x[1]) if str(x[1] or "").isdigit() else 0,
            )
        )
        folios[s] = rows
    zu, cids_all = {}, set()
    for r in zeilen:
        s, f = sid[r["quelle"]], r["id"].split("_")[0]
        rows = folios[s]
        idx = [i for i, x in enumerate(rows) if x[0] == f]
        cids = [rows[i][2] for i in idx]
        if idx and idx[0] > 0:
            cids.insert(0, rows[idx[0] - 1][2])  # Überlauf vom Vorfolio
        zu[f"{r['quelle']}/{r['id']}"] = list(dict.fromkeys(cids))
        cids_all.update(cids)
    fass = defaultdict(list)
    for cid in cids_all:
        for s, v in db.execute(
            "select source_id, volpiano from chants where cantus_id=? and volpiano is not null",
            (cid,),
        ):
            t = toene(v)
            if s not in AUSSCHLUSS and len(t) >= 4:
                fass[cid].append([s, t])
    out = {"zeilen": zu, "fassungen": fass, "quelle_von": sid}
    (HERE / "daten" / "konkordanz" / "konkordanzen.json").write_text(json.dumps(out))
    # Kennzahlen: je Zeile Fassungen aus anderen Quellen
    lines = [
        "# Melodie-RAG: Konkordanzdaten",
        "",
        "`skripte/konkordanz_daten.py`, Ausgabe `daten/konkordanz/konkordanzen.json` (nicht versioniert). Je Testzeile die Gesänge ihres Folios (plus letzter Gesang des Vorfolios) und deren Fassungen aus anderen Quellen (eigene Quelle und San Pietro B.79 ausgeschlossen).",
        "",
        "| Testmenge | Zeilen | Gesänge je Zeile (Median) | Zeilen mit mindestens einer fremden Fassung | fremde Fassungen je Zeile (Median) |",
        "|---|---|---|---|---|",
    ]
    test_keys = {f"{r['quelle']}/{r['id']}" for r in zeilen if r["split"] == "test"}
    val_keys = {f"{r['quelle']}/{r['id']}" for r in zeilen if r["split"] == "val"}
    for name, sel in [
        ("feste Testmenge (11 Quellen)", lambda k: k in test_keys),
        ("feste Validierung (11 Quellen)", lambda k: k in val_keys),
        ("A-KN 1012, alle Zeilen", lambda k: k.startswith("a-kn-1012/")),
    ]:
        ks = [k for k in zu if sel(k)]
        n_g = [len(zu[k]) for k in ks]
        n_f = []
        for k in ks:
            own = sid[k.split("/")[0]]
            n_f.append(sum(1 for c in zu[k] for s, _ in fass.get(c, []) if s != own))
        lines.append(
            f"| {name} | {len(ks)} | {statistics.median(n_g):.0f} | {sum(x > 0 for x in n_f) / len(ks):.0%} | {statistics.median(n_f):.0f} |"
        )
    lines += [
        "",
        f"{len(fass)} Gesänge mit Fassungen; Quellen je Gesang (Median) {statistics.median([len({s for s, _ in v}) for v in fass.values()]):.0f}.",
    ]
    (HERE / "docs" / "konkordanz_daten.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    assert toene("1---d-E--f4---g") == "def"
    main()
