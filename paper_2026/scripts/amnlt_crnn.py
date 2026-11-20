"""AMNLT (Einsiedeln, Salzinnes, Solesmes) als CRNN-Index aus denselben Zielen, auf denen LightOn trainiert.

    $TORCH -I skripte/amnlt_crnn.py          # schreibt daten/amnlt_crnn/<Datensatz>/music/index.jsonl
    $TORCH -I skripte/amnlt_crnn.py test     # Selbsttest der Umrechnung

Quelle: LightOnOCR-3-1B/daten/systeme.jsonl (5.602 Systeme, gemeinsame Kodierung, AMNLT-Splits wie veröffentlicht).
Umrechnung: `convert` aus amnlt_solesmes_export.py (gabc-Buchstabe -> Lage, Schlüssel, Striche -> L, Silbengruppen
durch `,`, Form- und Custoszeichen entfallen; Rückrechnung geprüft). Vorzeichen (`x` ♭, `y` Auflösung) werden
als `b_`/`na_`-Token behalten. Systeme mit anderem Ausschlussgrund (z. B. Vorzeichen auf einer Lage außerhalb des
Vokabulars) fallen aus train und val. Im Test bleiben alle Systeme: gemessen wird dort nicht am Label, sondern
an der Tonfolge des gabc-Ziels (Tonlagen-SER wie im LightOn-Artikel, metriken.py); nicht umrechenbare Testsysteme
tragen nur ihren Anfangsschlüssel als Platzhalter-Label (`ok` = false), damit der Datenlader sie liest.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amnlt_solesmes_export import convert  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
LIGHTON = HERE / "LightOnOCR-3-1B"
AUS = HERE / "daten/amnlt_crnn"
# Nur Token, die Decoder (vokabular.json) und CTC-Kopf (volpiano3_musici2w.json) kennen; sonst bricht der Datenlader ab
# (b_z-1 fehlt im CTC-Vokabular, MUSICA-Job 1957235).
VOK = set(json.load((HERE.parent / "trainingsdaten/vokabular.json").open())) & set(
    json.load(
        (
            Path(os.environ.get("OMR_CLI", "/home/rklug/projects_2026/omr_cli"))
            / "vocab/volpiano3_musici2w.json"
        ).open()
    ).values()
)
SPLIT = {"train": "train", "validation": "val", "test": "test"}


def zeile(z, basis):
    r = convert(z["ziel"], VOK, vorzeichen=True)
    ok = r["roundtrip_ok"]
    bild = LIGHTON / z["bild"]  # logischer Pfad: daten/ ist lokal ein Symlink auf D:
    b, h = z.get("b"), z.get("h")
    if not (b and h):
        b, h = Image.open(bild).size
    label = " ".join(r["tokens"]) if ok else (r["schluessel"] or "c4")
    return {
        "id": z["id"],
        "source": z["quelle"],
        "split": SPLIT[z["split"]],
        "image": os.path.relpath(bild, basis),
        "bbox": [0, 0, b, h],
        "label": label,
        "ok": ok,
        "grund": ";".join(sorted(r["reasons"])),
    }


def main():
    alle = [json.loads(z) for z in (LIGHTON / "daten/systeme.jsonl").open()]
    for ds in ("Einsiedeln", "Salzinnes", "Solesmes"):
        basis = AUS / ds
        zs = [zeile(z, basis) for z in alle if z["quelle"] == ds]
        zs = [z for z in zs if z["ok"] or z["split"] == "test"]
        (basis / "music").mkdir(parents=True, exist_ok=True)
        (basis / "music/index.jsonl").write_text(
            "".join(json.dumps(z, ensure_ascii=False) + "\n" for z in zs)
        )
        fehlt = [z["image"] for z in zs if not (basis / z["image"]).is_file()]
        assert not fehlt, fehlt[:3]
        print(ds, Counter((z["split"], z["ok"]) for z in zs))


def selbsttest():
    r = convert(
        "(<m>c<m>4) de(<m>g<m>h) us(<m>g<m>V<m>f) (<m>:) me(<m>e<m>f) (<m>f<m>+)", VOK
    )
    assert r["roundtrip_ok"] and r["tokens"] == "c4 z2 l3 , z2 l2 L z1 l2".split(), r
    r = convert("(<m>f<m>3) a(<m>i<m>x<m>i)", VOK)  # ohne Schalter: ausgeschlossen
    assert not r["roundtrip_ok"] and "vorzeichen" in r["reasons"], r
    r = convert("(<m>f<m>3) a(<m>i<m>x<m>i) b(<m>i<m>y<m>i)", VOK, vorzeichen=True)
    assert r["roundtrip_ok"] and r["tokens"] == "f3 b_z3 z3 , na_z3 z3".split(), r
    assert r["pitches"] == ["i", "i"], r
    print("Selbsttest in Ordnung")


if __name__ == "__main__":
    selbsttest() if sys.argv[1:] == ["test"] else main()
