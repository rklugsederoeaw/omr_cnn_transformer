# OMR für gregorianische Notenzeilen

Dieses Repo enthält eine eigenständig nutzbare Fassung des CRNN/Transformer-Modells `hybrid_hybrid_general_v10`, das Melodiemodell `decoder_lm_diff`, die für die Konkordanzbewertung benötigten CantusDB-Felder und zwei Layoutmodelle. Die Trainingsbilder und internen Forschungsdokumente sind für die Anwendung nicht erforderlich und liegen nicht hier.

## Installation

Python 3.12 oder neuer. Eine passende PyTorch-Installation für CPU oder NVIDIA-GPU wird benötigt.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Notenzeile lesen

Der zuverlässige Eingabetyp ist eine **zugeschnittene Notenzeile** (JPG oder PNG). Die Ausgabe ist JSON mit geometrischen Notentokens. `gelesen` kombiniert Bildmodell und Melodiemodell. `ohne_vorwissen` verwendet nur CTC und Attention.

```bash
.venv/bin/python -m omr.predict examples/001r_01_with_clef.jpg
```

Mit bekannter CantusDB-Quell-ID und Folio wird zusätzlich die Konkordanzbewertung aktiviert:

```bash
.venv/bin/python -m omr.predict examples/001r_01_with_clef.jpg --source-id 123667 --folio 001r
```

Für eine Notenzeile auf einem größeren Bild kann `--bbox X1 Y1 X2 Y2` angegeben werden. Ohne `--source-id` und `--folio` ist keine CantusDB-Abfrage nötig. Die kompakte Datenbank `weights/cantus_rag.sqlite` ist nur für diese optionale Bewertung nötig.

## Ganze Seite

`omr.page` sucht Notationsregionen mit YOLO und liest jede gefundene Region. Das ist eine automatische Vorschau; die erkannten Ausschnitte sollten geprüft und bei Bedarf als einzelne Zeilen mit `--bbox` gelesen werden.

```bash
.venv/bin/python -m omr.page seite.jpg --layout handschrift
.venv/bin/python -m omr.page seite.jpg --layout druck
```

`--source-id` und `--folio` sind auch hier möglich. Auf einer GPU wird CUDA automatisch genutzt; mit `--device cpu` kann die CPU erzwungen werden. Die JSON-Ausgabe enthält die Koordinaten jeder Region.

Die mitgelieferte Beispielzeile kann geprüft werden:

```bash
.venv/bin/python -m omr.check
```

## Inhalt und Herkunft

- `omr/model.py`: CRNN/Transformer-Architektur und Beam-Suche, aus den lokalen Projektdateien `skripte/hybrid.py`, `skripte/decoder_lm.py`, `skripte/joint.py` und `omr_cli/src/core/models/crnn.py` auf die Inferenz reduziert.
- `omr/predict.py`: Vorverarbeitung, Kandidaten und gemeinsame CTC/Attention/Melodiemodell-Bewertung.
- `omr/rag.py`: Konkordanzbewertung wie in `skripte/rag_bewertung.py`, `rag_bewertung2.py`, `rag_bewertung3.py` und `lesen_rag.py`.
- `omr/page.py`: optionale Layout-Erkennung. Die Layoutgewichte stammen aus `modelle/layout/{handschrift,druck}_s1/weights/best.pt`.
- `weights/hybrid_hybrid_general_v10.pt`: OMR-Gewichte (Stand 23.10.2026), inklusive Token `c5`.
- `weights/decoder_lm_diff.pt`: Melodiemodell mit Psalmdifferenzen; `weights/decoder_lm_vocab.json` und `weights/volpiano3_musici2w.json` definieren die Tokenreihenfolge.
- `weights/cantus_rag.sqlite`: nur `source_id`, `folio`, `sequence`, `cantus_id`, `volpiano` aus der CantusDB, ohne übrige Forschungsdaten.

Die Lesekette ist für vorhandene Gewichte gedacht. Der vollständige Trainingskorpus und die Trainingsskripte gehören nicht zu diesem Anwendungspaket.
