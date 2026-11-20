"""Vortrainingskorpus für einen Volpiano-Decoder im Zielformat des CRNN (docs/decoder_plan.md).

CantusDB-Volpiano aller Gesänge mit voller Melodie (ohne Incipit-Quellen, ohne San Pietro B.79), ohne die Gesänge der
Validierungs- und Testfolios des CRNN (plus letzter Gesang des vorangehenden Folios), an 7 in Zeilen geschnitten (lange
Gesänge ohne 7 an Wortgrenzen nach der Zeilenlängenverteilung der Ground Truth), je Zeile bis zu K plausible Schlüssel
(wenigste Positionen außerhalb des Systems, gewichtet nach der Schlüsselverteilung der Ground Truth), umgerechnet mit
scripts/volpiano_geo + scripts/align.export_tokens. Aufruf aus dem Projektordner:
nice -n 15 ~/miniconda3/envs/ai-base/bin/python3 -I skripte/decoder_korpus.py
"""

import csv
import hashlib
import json
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from align import export_tokens, geo_with_clefs  # noqa: E402
from volpiano_geo import pos_num, to_geo  # noqa: E402

K = 2  # Schlüsselvarianten je Zeile
import os  # noqa: E402

HOLDOUT = {
    int(x) for x in os.environ.get("HOLDOUT", "").split(",") if x
}  # Quellen ganz auslassen (Holdout-Experimente)
AUSSCHLUSS_QUELLEN = {123661} | HOLDOUT  # San Pietro B.79, altrömisch
CLEFS = [f"{c}{n}" for c in "cf" for n in range(1, 5)]
VOCAB = set(json.load((ROOT / "trainingsdaten" / "vokabular.json").open()))
OUT = (
    HERE / "daten" / ("decoder_korpus" + "".join(f"_ohne{s}" for s in sorted(HOLDOUT)))
)
POS = re.compile(r"^[lz]-?\d+$")


def folio_key(f):
    return [
        (0, int(x), "") if x.isdigit() else (1, 0, x)
        for x in re.findall(r"\d+|\D+", f or "")
    ]


def leak_set(db):
    """(source_id, node_id) der Gesänge auf Validierungs- und Testfolios des CRNN plus Überlauf vom Vorfolio."""
    sid_of = {
        json.load(p.open())["slug"]: json.load(p.open())["source_id"]
        for p in (ROOT / "sources").glob("*.json")
    }
    folios = defaultdict(set)
    for r in csv.DictReader(
        (ROOT / "trainingsdaten" / "zeilen.tsv").open(), delimiter="\t"
    ):
        if r["split"] in ("val", "test"):
            folios[sid_of[r["quelle"]]].add(r["id"].split("_")[0])
    out = set()
    for sid, fs in folios.items():
        rows = db.execute(
            "select node_id, folio, sequence from chants where source_id=?", (sid,)
        ).fetchall()
        rows.sort(
            key=lambda x: (
                folio_key(x[1]),
                int(x[2]) if str(x[2] or "").isdigit() else 0,
            )
        )
        for i, (nid, f, _) in enumerate(rows):
            if f in fs:
                out.add(nid)
                if (
                    i and rows[i - 1][1] != f
                ):  # letzter Gesang des Vorfolios läuft oft über
                    out.add(rows[i - 1][0])
    return out, sum(map(len, folios.values()))


def note_words(v):
    return [w for w in v.split("---") if re.search("[a-sA-S]", w)]


def full_melody(v, text):
    nw = len(note_words(v.split("4")[0]))
    tw = len((text or "").split())
    return nw >= 4 and (tw == 0 or nw >= 0.7 * tw)


def segments(v, rng, lengths):
    """Zeilen: an 7 schneiden; ohne 7 lange Gesänge an Wortgrenzen auf GT-typische Längen kürzen."""
    v = v.strip()
    v = v[1:] if v.startswith("1") else v
    parts = [p.strip("-") for p in re.split(r"7+", v)]
    out = []
    for p in parts:
        if len(re.findall("[a-sA-S]", p)) <= 60 or "7" in v:
            out.append(p)
            continue
        cur, n, target = [], 0, rng.choice(lengths)
        for w in p.split("---"):
            cur.append(w)
            n += len(re.findall("[a-sA-S]", w))
            if n >= target:
                out.append("---".join(cur))
                cur, n, target = [], 0, rng.choice(lengths)
        if cur:
            out.append("---".join(cur))
    return [s for s in out if re.search("[a-sA-S]", s)]


def clef_cost(seg, clef):
    toks = [t for t in to_geo(seg, clef) if POS.match(t.removesuffix("_liq"))]
    nums = [pos_num(t.removesuffix("_liq")) for t in toks]
    return sum(
        max(0, -1 - n) + max(0, n - 7) for n in nums
    ), toks  # im System: z-1 (-1) bis z4 (7)


def choose_clefs(seg, rng, prior):
    costs = {c: clef_cost(seg, c) for c in CLEFS}
    best = min(c for c, _ in costs.values())
    ok = [
        c
        for c in CLEFS
        if costs[c][0] == best
        and all(t.removesuffix("_liq") in VOCAB for t in costs[c][1])
    ]
    picked = []
    while ok and len(picked) < K:
        c = rng.choices(ok, weights=[prior.get(x, 1) for x in ok])[0]
        picked.append(c)
        ok.remove(c)
    return picked, best


def main():
    rng = random.Random(0)
    db = sqlite3.connect(f"file:{ROOT / 'cantus_database.sqlite'}?mode=ro", uri=True)
    gt = list(
        csv.DictReader((ROOT / "trainingsdaten" / "zeilen.tsv").open(), delimiter="\t")
    )
    prior = Counter()
    lengths = []
    for r in gt:
        t = r["label"].split()
        if t:
            eff = next(
                (p for p in t[0].split("/") if p[0] in "cf" and p[1:].isdigit()), None
            )
            if eff:
                prior[eff] += 1
        lengths.append(max(10, sum(1 for x in t if POS.match(x))))
    leak, n_fol = leak_set(db)
    st = Counter()
    OUT.mkdir(parents=True, exist_ok=True)
    fh = (OUT / "korpus.jsonl").open("w")
    for nid, sid, cid, genre, mode, v, text in db.execute(
        "select node_id, source_id, cantus_id, genre, mode, volpiano, fulltext_standardized from chants where volpiano is not null"
    ):
        st["gesänge"] += 1
        if sid in AUSSCHLUSS_QUELLEN:
            st["ausgeschlossen altrömisch"] += 1
            continue
        if nid in leak:
            st["ausgeschlossen Testleck"] += 1
            continue
        if not full_melody(v, text):
            st["kein volle Melodie"] += 1
            continue
        st["gesänge im Korpus"] += 1
        lm_split = (
            "val"
            if int(hashlib.md5((cid or str(nid)).encode()).hexdigest(), 16) % 20 == 0
            else "train"
        )
        for i, seg in enumerate(segments(v, rng, lengths)):
            clefs, cost = choose_clefs(seg, rng, prior)
            if not clefs:
                st["zeile ohne passenden Schlüssel"] += 1
                continue
            st["zeilen"] += 1
            st["zeilen mit Hilfslinien außerhalb z-1..z4"] += cost > 0
            for c in clefs:
                toks = export_tokens(geo_with_clefs(seg, [(0, c)]))
                if not toks or any(t not in VOCAB for t in toks):
                    st["sequenz verworfen (Vokabular)"] += 1
                    continue
                st["sequenzen"] += 1
                st["tokens"] += len(toks)
                fh.write(
                    json.dumps(
                        {
                            "tokens": " ".join(toks),
                            "node_id": nid,
                            "source_id": sid,
                            "cantus_id": cid,
                            "genre": genre,
                            "mode": mode,
                            "zeile": i,
                            "clef": c,
                            "split": lm_split,
                        }
                    )
                    + "\n"
                )
    fh.close()
    lines = [
        "# Decoder-Vortrainingskorpus",
        "",
        f"Erzeugt mit `skripte/decoder_korpus.py`, Ausgabe `daten/decoder_korpus/korpus.jsonl` (nicht versioniert). Testleck: {n_fol} Validierungs- und Testfolios des CRNN, {len(leak)} Gesänge ausgeschlossen. Bis zu {K} Schlüssel je Zeile, gewichtet nach der Ground Truth ({', '.join(f'{c} {n}' for c, n in prior.most_common())}). Sprachmodell-Validierung: 5 % der Cantus-IDs (alle Fassungen gemeinsam).",
        "",
        "| Größe | Anzahl |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v:,} |".replace(",", ".") for k, v in st.items()]
    (HERE / "docs" / f"{OUT.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    assert folio_key("010v") > folio_key("009r")
    assert full_melody("1---d-e---f---g--a---h", "a b c d") and not full_melody(
        "1---d-e", "a b c"
    )
    main()
