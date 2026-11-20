"""Lesekette mit Konkordanzbewertung (Melodie-RAG) für beliebige Zeilen einer Quelle.

Empfohlene Kette: Hybrid (korrigiertes LSTM, neutraler Decoder), gemeinsame CTC/Attention-Decodierung, unverändertes
Melodiemodell als dritte Stimme, Konkordanzbewertung mit Mehrheitslage (docs/rag.md). Nutzt die Bausteine des Klons
unverändert: Kandidaten über skripte/rag_kandidaten2.py (als Unterprozess in der Torch-Umgebung), Konkordanzdaten nach
der Logik von skripte/konkordanz_daten.py (folio_key, toene, AUSSCHLUSS; hier für beliebige (source_id, folio)),
Bewertung über rag_bewertung2.prepare und rag_bewertung3.arrays; die Auswahl ist die Formel aus rag_bewertung3.evaluate
(tau unendlich, Mehrheitslage an).

Eingaben: Index im Format des Pipeline-Index (image, bbox, split, source, id; label optional), CantusDB-source_id der
Quelle, Hybrid-Modell, Melodiemodell, Gewichte beta, mu, rho (Standard 0,0 / 0,1 / 1,0 aus dem Holdout A-KN 1012).
Ausgabe JSONL je Zeile: key, gelesen (gewählte Tokenfolge), ohne_vorwissen (0,3·CTC + 0,7·Att + beta·Länge),
schluessel_geaendert (Anfangsschlüssel weicht von der Lesung ohne Vorwissen ab), geaendert (Lesung weicht ab).
Mit Labels im Index zusätzlich Token-SER und Melodie-SER (wie rag_bewertung3.evaluate).

Aufruf (aus natural_music_processing, Umgebung .venv_helsen):
.venv_helsen/bin/python -I skripte/lesen_rag.py --index <index.jsonl> --source-id 123613 --modell hybrid_scratch_loso1012 \
    --lm decoder_lm_ohne123613 --name lesen_loso1012 [--split test] [--beta 0 --mu 0.1 --rho 1]
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import konkordanz_daten as K  # noqa: E402
import rag_bewertung2 as B  # noqa: E402
import rag_bewertung3 as C  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
TORCH_PY = os.environ.get(
    "TORCH_PY", "/home/rklug/projects_2026/omr_cli/.venv/bin/python"
)  # Cluster: Python der venv


def key_of(image):
    """Schlüssel wie in rag_kandidaten2.py: <Ordner über images>/<Bildname>."""
    p = Path(image)
    return f"{p.parent.parent.name}/{p.stem}"


def baue_index(index, split, name):
    """Arbeitsindex: gewählte Zeilen als Test, Bildpfade absolut, fehlende Labels als ',' (nur für den Datenlader)."""
    src = Path(index)
    rows = []
    for line in src.open():
        e = json.loads(line)
        if split and e.get("split") != split:
            continue
        img = Path(e["image"])
        e["image"] = str(
            img if img.is_absolute() else (src.parent.parent / img).resolve()
        )
        e["label"] = e.get("label") or ","
        e["split"] = "test"
        rows.append(e)
    ziel = HERE / "daten" / "lesen_rag" / name / "music"
    ziel.mkdir(parents=True, exist_ok=True)
    (ziel / "index.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in rows)
    )
    return ziel.parent, rows


def konkordanzen(rows, source_id=None):
    """Struktur wie daten/konkordanz/konkordanzen.json, für die Zeilen einer Quelle: Gesänge ihres Folios (plus letzter
    Gesang des Vorfolios) und deren Fassungen aus anderen Quellen (eigene und San Pietro B.79 ausgeschlossen). Ohne
    source_id wird sie je Zeile aus sources/<slug>.json bestimmt (feste Testmenge mit elf Quellen)."""
    db = sqlite3.connect(
        f"file:{HERE.parent / 'cantus_database.sqlite'}?mode=ro", uri=True
    )
    sid_of = {
        json.load(q.open())["slug"]: json.load(q.open())["source_id"]
        for q in (HERE.parent / "sources").glob("*.json")
    }
    cache = {}

    def chants_of(s):
        if s not in cache:
            c = db.execute(
                "select folio, sequence, cantus_id from chants where source_id=? and cantus_id is not null",
                (s,),
            ).fetchall()
            c.sort(
                key=lambda x: (
                    K.folio_key(x[0]),
                    int(x[1]) if str(x[1] or "").isdigit() else 0,
                )
            )
            cache[s] = c
        return cache[s]

    zu, alle, quelle_von = {}, set(), {}
    for e in rows:
        k = key_of(e["image"])
        s = source_id or sid_of[k.split("/")[0]]
        quelle_von[k.split("/")[0]] = s
        chants = chants_of(s)
        f = e["id"].split("_")[0]
        idx = [i for i, x in enumerate(chants) if x[0] == f]
        cids = [chants[i][2] for i in idx]
        if idx and idx[0] > 0:
            cids.insert(0, chants[idx[0] - 1][2])
        zu[k] = list(dict.fromkeys(cids))
        alle.update(cids)
    fass = defaultdict(list)
    for cid in alle:
        for s, v in db.execute(
            "select source_id, volpiano from chants where cantus_id=? and volpiano is not null",
            (cid,),
        ):
            t = K.toene(v)
            if s not in K.AUSSCHLUSS and len(t) >= 4:
                fass[cid].append([s, t])
    return {"zeilen": zu, "fassungen": fass, "quelle_von": quelle_von}


def waehle(s0, lm, ragb, cands, mu, rho, sperre, fest=False):
    """Index der gewählten Lesart. Ohne sperre = rag_bewertung3.evaluate (tau unendlich, Mehrheitslage). Mit sperre
    bestimmen Bild und Konkordanz den Anfangsschlüssel ohne Melodiemodell; das Melodiemodell wählt nur unter den
    Kandidaten mit diesem Schlüssel (docs/hybrid_vergleich.md: mehr Schlüsseltreffer, Melodie-SER nicht besser).
    Mit fest bleibt der Anfangsschlüssel der Lesung ohne Vorwissen; Melodiemodell und Konkordanzen wählen nur unter
    den Kandidaten mit diesem Schlüssel (für Quellen, die das Modell kennt: Bildprüfung 19.11.2026,
    docs/arxiv/neufassung/lehren_schluessel.md)."""
    s = s0 + mu * lm + rho * ragb
    if fest:
        k = cands[int(np.argmax(s0))][:1]
        return int(np.argmax(np.where([c[:1] == k for c in cands], s, -np.inf)))
    if not sperre:
        return int(np.argmax(s))
    k = cands[int(np.argmax(s0 + rho * ragb))][:1]
    return int(np.argmax(np.where([c[:1] == k for c in cands], s, -np.inf)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--index", required=True)
    ap.add_argument(
        "--source-id",
        type=int,
        default=None,
        help="CantusDB-source_id; ohne Angabe je Zeile aus sources/<slug>.json",
    )
    ap.add_argument("--modell", required=True)
    ap.add_argument("--lm", required=True)
    ap.add_argument(
        "--name",
        required=True,
        help="Arbeitsname (Ordner daten/lesen_rag/<name>, Kandidaten-TAG)",
    )
    ap.add_argument(
        "--split",
        default="",
        help="nur Zeilen dieses Splits aus dem Index (leer = alle)",
    )
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument(
        "--lm-schluessel-sperre",
        action="store_true",
        help="Melodiemodell darf den Anfangsschlüssel nicht ändern (Konkordanz weiterhin); auf ungesehenen Büchern nicht geprüft",
    )
    ap.add_argument(
        "--schluessel-fest",
        action="store_true",
        help="Vorwissen darf den Anfangsschlüssel der Lesung ohne Vorwissen nicht ändern (Quellen, die das Modell kennt)",
    )
    a = ap.parse_args()
    data, rows = baue_index(a.index, a.split, a.name)
    tag = a.name
    kand = HERE / "daten" / "konkordanz" / f"kandidaten_{tag}_test.jsonl"
    if not kand.exists():
        env = dict(
            os.environ,
            DATA=str(data),
            MODELL=a.modell,
            LM=a.lm,
            TAG=tag,
            OMP_NUM_THREADS="1",
        )
        subprocess.run(
            [TORCH_PY, "-I", str(HERE / "skripte" / "rag_kandidaten2.py")],
            env=env,
            check=True,
            cwd=HERE,
        )
    konk = konkordanzen(rows, a.source_id)
    prep = B.prepare(tag, "test", konk)
    arr = C.arrays(prep)
    out = HERE / "daten" / "lesen_rag" / a.name / "lesungen.jsonl"
    n_s = n_g = 0
    gewaehlt = []
    with out.open("w") as fh:
        for r, x in zip(prep, arr):
            s0 = x["base"] + a.beta * x["ln"]
            i0 = int(np.argmax(s0))
            i = waehle(
                s0,
                x["lm"],
                x["ragb"],
                r["cands"],
                a.mu,
                a.rho,
                a.lm_schluessel_sperre,
                a.schluessel_fest,
            )
            gewaehlt.append(i)
            c, c0 = r["cands"][i], r["cands"][i0]
            sg = bool(c and c0 and c[0] != c0[0])
            n_s += sg
            n_g += c != c0
            fh.write(
                json.dumps(
                    {
                        "key": r["key"],
                        "gelesen": c,
                        "ohne_vorwissen": c0,
                        "schluessel_geaendert": sg,
                        "geaendert": c != c0,
                        # Attention-Punktzahl je Token der gewählten Lesart; niedrig = unsicher (skripte/rueckblick_konfidenz.py)
                        "konfidenz": round(r["att"][i] / max(len(c), 1), 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"{len(prep)} Zeilen -> {out}; Lesung geändert: {n_g}, davon Anfangsschlüssel geändert: {n_s}"
    )
    if all(e["label"] != "," for e in rows):
        t0, m0 = C.evaluate(arr, a.beta, 0.0, 0.0, 1e9, True)
        t1, m1 = C.evaluate(arr, a.beta, a.mu, a.rho, 1e9, True)
        if (
            a.lm_schluessel_sperre or a.schluessel_fest
        ):  # evaluate kennt die Sperren nicht
            t1 = (
                100
                * sum(x["ed"][i] for x, i in zip(arr, gewaehlt))
                / sum(x["glen"] for x in arr)
            )
            m1 = (
                100
                * sum(x["med"][i] for x, i in zip(arr, gewaehlt) if x["mlen"])
                / sum(x["mlen"] for x in arr)
            )
        print(
            f"Kontrolle (beta {a.beta}, mu {a.mu}, rho {a.rho}): ohne Vorwissen Token-SER {t0:.2f} % / Melodie-SER {m0:.2f} %; mit Konkordanzen {t1:.2f} % / {m1:.2f} %"
        )


if __name__ == "__main__":
    assert key_of("/x/export/a-kn-1012/images/012r_03.jpg") == "a-kn-1012/012r_03"
    _k = [["c3", "l1"], ["c3", "l2"], ["f2", "l1"]]
    _s, _l, _r = np.array([0.0, -0.1, -0.5]), np.array([-5.0, -1.0, -0.1]), np.zeros(3)
    assert waehle(_s, _l, _r, _k, 1.0, 1.0, False) == 2  # LM wechselt den Schlüssel
    assert waehle(_s, _l, _r, _k, 1.0, 1.0, True) == 1  # mit Sperre nur die Note
    _r2 = np.array([0.0, 0.0, 5.0])  # Konkordanz zieht zum F-Schlüssel
    assert waehle(_s, _l, _r2, _k, 1.0, 1.0, True) == 2  # die LM-Sperre lässt das zu
    assert (
        waehle(_s, _l, _r2, _k, 1.0, 1.0, False, True) == 1
    )  # fest: Schlüssel bleibt, nur die Note wechselt
    main()
