"""Lesekette in vier Stufen für ein Modell und eine Testmenge (Messreihen M1 bis M4, docs/arxiv/neufassung/messplan.md).

    $TORCH -I skripte/lesen_reihe.py --index <index.jsonl> --split test --modell <hybrid_...> --lm <decoder_lm...>
        --name <name> [--source-id <id>]

Ruft lesen_rag.py dreimal mit demselben Namen auf; die Kandidaten (Beam 12, Schlüsselvarianten) entstehen beim ersten
Aufruf und werden danach wiederverwendet. Gewichte fest, auf der Validierung der festen Aufteilung gewählt
(docs/rag.md, „fest2 B“): beta 0,5; voll mu 0,05 und rho 0,5; nur Melodiemodell rho 0; nur Konkordanzen mu 0.
Ausgabe daten/lesen_rag/<name>/lesungen_<stufe>.jsonl für voll, nur_lm, nur_konk; die Lesung ohne Vorwissen
(gemeinsame Decodierung) steht in jeder Datei im Feld ohne_vorwissen. --source-id -1: keine Konkordanzen
(Quelle nicht in der CantusDB oder ohne Foliozuordnung).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
STUFEN = {"voll": ("0.05", "0.5"), "nur_lm": ("0.05", "0"), "nur_konk": ("0", "0.5")}


def main(args):
    name = args[args.index("--name") + 1]
    ziel = HERE / "daten/lesen_rag" / name
    py = os.environ.get("TORCH_PY", sys.executable)
    for stufe, (mu, rho) in STUFEN.items():
        cmd = [py, "-I", str(HERE / "skripte/lesen_rag.py"), *args]
        cmd += ["--beta", "0.5", "--mu", mu, "--rho", rho]
        print("===", stufe, " ".join(cmd[3:]), flush=True)
        subprocess.run(cmd, check=True, cwd=HERE)
        shutil.copy(ziel / "lesungen.jsonl", ziel / f"lesungen_{stufe}.jsonl")


if __name__ == "__main__":
    main(sys.argv[1:])
