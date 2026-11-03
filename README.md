# Optical Music Recognition for Gregorian chant

This repository provides a standalone inference package for the `hybrid_hybrid_general_v10` CRNN/Transformer model. It includes the `decoder_lm_diff` melody model, the CantusDB fields needed for concordance scoring, and two page layout models. Training images and internal research documents are not needed to run inference and are not included.

## Installation

Use Python 3.12 or newer and install a PyTorch build suitable for your CPU or NVIDIA GPU.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Read a notation line

The most reliable input is an image cropped to a **single notation line** (JPG or PNG). The command prints JSON containing geometric music tokens. `gelesen` combines the image model and melody model; `ohne_vorwissen` uses only the image model's CTC and attention scores.

```bash
.venv/bin/python -m omr.predict examples/001r_01_with_clef.jpg
```

If the CantusDB source ID and folio are known, add concordance scoring:

```bash
.venv/bin/python -m omr.predict examples/001r_01_with_clef.jpg --source-id 123667 --folio 001r
```

Use `--bbox X1 Y1 X2 Y2` to select a notation line within a larger image. The compact `weights/cantus_rag.sqlite` database is used only when both `--source-id` and `--folio` are supplied.

The token sequence uses `l1`, `l2`, etc. for staff lines, `z1`, `z2`, etc. for spaces, clef tokens such as `f2`, and `,` as the separator used by the model. The returned `konfidenz` is an attention log score per token; higher values indicate a better model fit. It is not a calibrated probability.

## Read a full page

`omr.page` detects notation regions with YOLO and reads each region. Treat these results as a preview: inspect the reported bounding boxes and reread individual lines with `--bbox` where needed.

```bash
.venv/bin/python -m omr.page page.jpg --layout handschrift
.venv/bin/python -m omr.page page.jpg --layout druck
```

The page command also accepts `--source-id` and `--folio`. CUDA is selected automatically when available; use `--device cpu` to force CPU inference. The JSON output includes each region's coordinates.

Check the bundled example against its reference transcription:

```bash
.venv/bin/python -m omr.check
```

## Contents and provenance

- `omr/model.py`: CRNN/Transformer architecture and beam search, adapted for inference from `skripte/hybrid.py`, `skripte/decoder_lm.py`, `skripte/joint.py`, and `omr_cli/src/core/models/crnn.py` in the source projects.
- `omr/predict.py`: image preprocessing, candidate generation, and combined CTC/attention/melody-model scoring.
- `omr/rag.py`: concordance scoring adapted from `skripte/rag_bewertung.py`, `rag_bewertung2.py`, `rag_bewertung3.py`, and `lesen_rag.py`.
- `omr/page.py`: optional page layout detection. Its weights come from `modelle/layout/{handschrift,druck}_s1/weights/best.pt`.
- `weights/hybrid_hybrid_general_v10.pt`: OMR model weights from 23 October 2026, including the `c5` token.
- `weights/decoder_lm_diff.pt`: melody model trained with psalm differentiae. The two vocabulary JSON files define the token order.
- `weights/cantus_rag.sqlite`: compact CantusDB extract containing only `source_id`, `folio`, `sequence`, `cantus_id`, and `volpiano`.

This package runs the supplied weights. The full training corpus and training scripts are outside its scope.
