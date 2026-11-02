"""Read a cropped chant staff with the trained CRNN/Transformer and melody LM."""
from __future__ import annotations

import argparse
import json
import re
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from omr.model import CRNN_OMR, Hybrid, MelodyLM, att_score, beam_search, lm_score

ROOT = Path(__file__).resolve().parent.parent
CLEF = re.compile(r"^[a-g]-?\d(/[a-g]\d)*$")


def load_models(device: torch.device, model_path: Path, lm_path: Path):
    vocab = json.loads((ROOT / "weights/volpiano3_musici2w.json").read_text())
    i2w = {int(k): v for k, v in vocab.items()}
    tokens = json.loads((ROOT / "weights/decoder_lm_vocab.json").read_text()) + ["c5"]
    i2w[len(i2w)] = "c5"
    blank = len(i2w)
    model = Hybrid(CRNN_OMR(blank + 1), None, blank + 1, len(tokens))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval().to(device)
    lm_ck = torch.load(lm_path, map_location="cpu", weights_only=True)
    lm = MelodyLM(len(tokens), **lm_ck["cfg"])
    state = lm_ck["state"]
    if len(lm_ck["vocab"]) + 1 == len(tokens):
        # The image model has one later token (c5), absent from the LM corpus.
        generator = torch.Generator().manual_seed(1)
        for key in ("tok.weight", "head.weight"):
            original = state[key]
            exemplar = original[lm_ck["vocab"].index("c4") : lm_ck["vocab"].index("c4") + 1]
            noise = torch.randn((1, original.shape[1]), generator=generator).to(original.dtype)
            state[key] = torch.cat((original, exemplar + 0.1 * original.std() * noise))
    elif lm_ck["vocab"] != tokens:
        raise ValueError("The melody model vocabulary does not match the image model")
    lm.load_state_dict(state)
    lm.eval().to(device)
    return model, lm, i2w, tokens, blank


def read_crop(image: Path, bbox: tuple[int, int, int, int] | None):
    page = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if page is None:
        raise FileNotFoundError(image)
    if bbox is None:
        bbox = (0, 0, page.shape[1] - 1, page.shape[0] - 1)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("The bounding box must have positive width and height")
    height = 128
    width = max(1, round(height * (x2 - x1) / (y2 - y1)))
    xs, ys = np.meshgrid(np.linspace(x1, x2, width, dtype=np.float32), np.linspace(y1, y2, height, dtype=np.float32))
    crop = cv2.remap(page, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x = torch.from_numpy(crop).float()[None, None] / 255.0
    return x, width // 4


@torch.no_grad()
def predict(image: Path, model: Hybrid, lm: MelodyLM, i2w: dict[int, str], tokens: list[str], blank: int, device: torch.device, bbox=None, beam=6, source_id=None, folio=None, cantus_db=None):
    x, frames = read_crop(image, bbox)
    x = x.to(device)
    fl = torch.tensor([frames], device=device)
    w2i = {token: index for index, token in enumerate(tokens)}
    bos, eos = w2i["<bos>"], w2i["<eos>"]
    h = model.encode(x)
    lpc = model.ctc(h).float().log_softmax(-1)
    beams = beam_search(model, h, fl, bos, eos, beam)[0]
    ctc = [t for t, _ in groupby(lpc[0, :frames].argmax(-1).tolist()) if t != blank]
    gd = [w2i[i2w[t]] for t in ctc]
    dec2ctc = {w2i[v]: k for k, v in i2w.items()}
    cands = [s for s, _ in beams if s and all(t in dec2ctc for t in s)]
    basis = cands[:3] + ([gd] if gd else [])
    if gd and gd not in cands:
        cands.append(gd)
    clefs = [w2i[v] for v in i2w.values() if CLEF.match(v)]
    for cand in basis:
        if cand and CLEF.match(tokens[cand[0]]):
            for clef in clefs:
                variant = [clef] + cand[1:]
                if variant not in cands:
                    cands.append(variant)
    if not cands:
        cands = [gd or [w2i[","]]]
    att = att_score(model, h, fl[0], cands, bos, eos).tolist()
    targets = [torch.tensor([dec2ctc[t] for t in cand] or [0], dtype=torch.long) for cand in cands]
    padded = nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)
    lp_rep = lpc[:, :frames].permute(1, 0, 2).expand(-1, len(cands), -1)
    ctc_scores = (-nn.functional.ctc_loss(lp_rep, padded, fl.repeat(len(cands)), torch.tensor([len(t) for t in targets], device=device), blank=blank, reduction="none", zero_infinity=True)).tolist()
    melody_scores = lm_score(lm, cands, bos, eos, device)
    base = [0.3 * c + 0.7 * a for c, a in zip(ctc_scores, att)]
    best_base = max(range(len(cands)), key=base.__getitem__)
    combined = [b + 0.1 * m for b, m in zip(base, melody_scores)]
    if source_id is not None and folio is not None:
        from omr.rag import score
        db = Path(cantus_db) if cantus_db else ROOT / "weights/cantus_rag.sqlite"
        if not db.is_file():
            raise FileNotFoundError(f"Concordance database missing: {db}")
        rag = score([[tokens[t] for t in c] for c in cands], best_base, source_id, folio, db)
        combined = [value + boost for value, boost in zip(combined, rag)]
    best = max(range(len(cands)), key=combined.__getitem__)
    return {"image": str(image), "rag_angewendet": source_id is not None and folio is not None, "gelesen": [tokens[t] for t in cands[best]], "ohne_vorwissen": [tokens[t] for t in cands[best_base]], "konfidenz": round(att[best] / max(len(cands[best]), 1), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="cropped notation line, JPG or PNG")
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="optional line box within a larger image")
    parser.add_argument("--model", type=Path, default=ROOT / "weights/hybrid_hybrid_general_v10.pt")
    parser.add_argument("--lm", type=Path, default=ROOT / "weights/decoder_lm_diff.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--source-id", type=int, help="CantusDB source ID; enables concordance scoring with --folio")
    parser.add_argument("--folio", help="folio of this notation line, e.g. 001r")
    parser.add_argument("--cantus-db", type=Path, help="override the bundled compact CantusDB extract")
    args = parser.parse_args()
    if (args.source_id is None) != (args.folio is None):
        parser.error("--source-id and --folio must be supplied together")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    torch.set_num_threads(4)
    model, lm, i2w, tokens, blank = load_models(device, args.model, args.lm)
    print(json.dumps(predict(args.image, model, lm, i2w, tokens, blank, device, args.bbox, args.beam, args.source_id, args.folio, args.cantus_db), ensure_ascii=False))


if __name__ == "__main__":
    main()
