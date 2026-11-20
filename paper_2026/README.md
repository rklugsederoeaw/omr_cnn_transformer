# Paper reproduction: joint CTC/attention decoding and repertoire priors (2026)

This folder accompanies the article *Joint CTC/Attention Decoding and Repertoire Priors for Staff-Level Recognition of Medieval Plainchant* (R. Klugseder, Austrian Academy of Sciences; arXiv identifier to be added). It contains the training, reading and evaluation scripts used for the article, the split indices without images, and the weights of the models of the main comparisons. The inference package in the repository root (`omr/`, `general_v10`) is a different, later production model and is not the model evaluated in the article.

## Contents

| Path | Content |
| --- | --- |
| `scripts/` | The working scripts of the study, unchanged (comments and messages in German). Mapping to the article below. |
| `scripts/musica/` | SLURM job frame and job waves for the MUSICA cluster (Austrian Scientific Computing). |
| `splits/fixed_split.jsonl.gz` | Index of the Cantus-aligned ground truth: image path, crop box, target sequence, split, source and staff id. The fixed test set has 3,084 staves. |
| `splits/general_ohne_pruef.jsonl.gz` | Training index of the CRNN of Section 6.4 (Cantus-aligned ground truth plus 22 reviewed Transkribus books, without the five held-out manuscripts). |
| `splits/comparison_known.jsonl.gz`, `comparison_unseen.jsonl.gz` | Shared test staves with the vision-language model (1,041 known, 200 unseen). |
| `splits/amnlt_*.jsonl.gz` | AMNLT staves converted to our encoding (Section 6.3). |
| `weights/` | See below. |
| `SHA256SUMS` | Checksums of weights and splits. |

The images are not included (see *Image rights*). The leave-one-source-out indices are generated from `fixed_split.jsonl.gz` by `scripts/loso_index.py`.

## Weights

| File | Article | Description |
| --- | --- | --- |
| `hybrid_hybrid_m1_s0.pt` | Table 5, seed 0 | Hybrid with decoder initialised from the melody model, fixed split |
| `hybrid_scratch_m1_s0.pt` | Table 5, seed 0 | Hybrid with neutral (randomly initialised) decoder |
| `hybrid_ctc_m1_s0.pt` | Table 5, seed 0 | CTC-only model |
| `hybrid_hybrid_m4g_s0.pt` | Table 10, seed 0 | CRNN of the comparison with LightOnOCR-3-1B, trained without the five held-out manuscripts |
| `decoder_lm.pt` | Sections 4.1–4.3 | Melody model (decoder initialisation and melody-model rescoring) |

Seeds 1 and 2, the fold models and the AMNLT models are available on request. The CTC vocabulary is `../weights/volpiano3_musici2w.json`; the decoder vocabulary is stored in `decoder_lm.pt`.

## Scripts and article sections

| Article | Scripts |
| --- | --- |
| 3.2 melody corpus, 4.1 melody model | `decoder_korpus.py`, `decoder_lm.py`, `volpiano_geo.py`, `align.py` |
| 4.1–4.2 model and training | `hybrid.py` (environment variables `MODUS`, `SEED`, `INIT`, `LM_INIT`, `DATA`) |
| 4.3 candidates, joint rescoring, priors | `joint.py`, `lesen_rag.py`, `lesen_reihe.py`, `rag_bewertung.py`, `rag_bewertung2.py`, `rag_bewertung3.py`, `konkordanz_daten.py` |
| 3.3 folds and test sets | `loso_index.py`, `general_ohne_pruef.py`, `m4_mengen.py`, `amnlt_crnn.py`, `amnlt_solesmes_export.py` |
| 6.1–6.2 intervals and clef transitions | `auswertung_vorwissen.py` (`GRUPPE=folio`: bootstrap over source–folio pairs) |
| 6.2 image checks | `pruefbogen_schluessel.py`, `pruefbogen_produktion.py`, `pruefbogen_auswerten.py`, `pruefbogen_produktion_auswerten.py` |
| 6.3–6.4 comparison and AMNLT | `vergleich_m4.py` with `metriken.py` and `vergleich_gt.py` from the LightOn project |
| 6.4 reading times | `lesezeit_crnn.py` |
| 6.5 error types | `fehlerarten_crnn.py` |
| Figures 1–2 | `abbildungen_artikel.py` |

The scripts expect the data loader and CRNN encoder of the ÖAW/PRAIG recogniser [ACDHxUA/CLI](https://github.com/ACDHxUA/CLI) (`omr_cli`, revision a1ae9e5), set with `OMR_CLI`; their default paths point to the author's machine and are overridden by environment variables (`OMR_CLI`, `DATA`, `LIGHTON_CODE`). Environment of the article: NVIDIA PyTorch container 25.04 (PyTorch 2.7.0a0+79aa17489c.nv25.04, NumPy 1.26.4, OpenCV 4.11.0, Lightning 2.6.6) with RapidFuzz 3.14.6 and transformers 4.46.3; statistics with Python 3.12.15, NumPy 2.5.3 and RapidFuzz 3.14.6. Job numbers and revisions are listed in the appendix of the article.

Example, intervals and clef transitions on the fixed test set from stored readings:

```bash
GRUPPE=folio python -I scripts/auswertung_vorwissen.py <fixed_split index.jsonl> <folder with lesungen_voll.jsonl>
```

## Using the repertoire priors

The article proposes a deployment heuristic: let the melody model and the concordance score change the initial clef only for manuscripts the model has not seen. For manuscripts in the training data, keep the clef of the reading without priors (`--lm-clef-lock` in the inference package, `--schluessel-fest` in `lesen_rag.py`). The heuristic is derived from image checks of changed clefs in four test sets and is not validated on further manuscripts.

## Licences

- Code in `scripts/`: Apache License 2.0 (see `../LICENSE`).
- Weights and split indices: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). The target sequences are derived from the Cantus Database (CC BY-NC-SA 4.0; cite as *Cantus Database*, directed by Debra Lacoste, https://cantusdatabase.org, data export of 16 February 2026) and from ground truth of the Austrian Academy of Sciences; the AMNLT staves come from the AMNLT datasets of PRAIG (CC BY-NC 4.0; Fuentes-Martínez et al., *Pattern Recognition* 170, 2026, art. 112094).

## Image rights of the training sources

The models were trained on staff crops of the following digitised manuscripts. The images themselves are not distributed here.

| Source | Holding institution | Images |
| --- | --- | --- |
| CH-E 611, CH-Fco Ms. 2, CH-P 18 | Stiftsbibliothek Einsiedeln; Franziskanerkloster Freiburg/Fribourg; Bibliothèque cantonale jurassienne, Porrentruy | e-codices, CC BY-NC |
| D-KA Aug. LX (Aug. perg. 60) with addendum | Badische Landesbibliothek Karlsruhe | public library, scientific use |
| D-KNd 1161 | Erzbischöfliche Diözesan- und Dombibliothek Köln | public library, scientific use |
| F-Pn lat. 12044 | Bibliothèque nationale de France (Gallica) | public library, scientific use |
| A-Wn Cod. 1799** | Österreichische Nationalbibliothek | public library, scientific use |
| A-KN 1010, 1012, 1018 | Stiftsbibliothek Klosterneuburg | Austrian Academy of Sciences |
| Transkribus books of Section 6.4 (twelve A-Gf volumes, A-BRLm P 33 and P 34, A-BRz fragment, A-LIktu 13, A-SF XI 409, A-VOR 259/1, A-Wda Cod. 3) | various Austrian holdings | Austrian Academy of Sciences |
| D-Mbs Clm 2766 | Bayerische Staatsbibliothek | CC BY-NC-SA 4.0 |
| I-BREmd (Taisten), I-SCass VII A 7 | Italian holdings, photographed by the author | Austrian Academy of Sciences |
| AMNLT: Einsiedeln, Salzinnes, Solesmes | PRAIG datasets on Hugging Face | CC BY-NC 4.0 (dataset licence) |
