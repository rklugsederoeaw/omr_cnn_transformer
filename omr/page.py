"""Detect notation regions on a page and read each region with the OMR model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from omr.predict import ROOT, load_models, predict

LAYOUT = {
    "handschrift": ROOT / "weights/layout_handschrift.pt",
    "druck": ROOT / "weights/layout_druck.pt",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="full page image")
    parser.add_argument("--layout", choices=LAYOUT, default="handschrift")
    parser.add_argument("--source-id", type=int, help="CantusDB source ID; supply with --folio for concordance scoring")
    parser.add_argument("--folio", help="page folio, e.g. 001r")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--beam", type=int, default=6)
    args = parser.parse_args()
    if (args.source_id is None) != (args.folio is None):
        parser.error("--source-id and --folio must be supplied together")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    torch.set_num_threads(4)
    detector = YOLO(str(LAYOUT[args.layout]))
    result = detector.predict(str(args.image), device=str(device), verbose=False)[0]
    boxes = []
    for box in result.boxes:
        if result.names[int(box.cls.item())] == "notation":
            bounds = tuple(round(float(v)) for v in box.xyxy[0].tolist())
            boxes.append(bounds)
    boxes.sort(key=lambda b: (b[1], b[0]))
    model, lm, i2w, tokens, blank = load_models(device, ROOT / "weights/hybrid_hybrid_general_v10.pt", ROOT / "weights/decoder_lm_diff.pt")
    rows = []
    for box in boxes:
        row = predict(args.image, model, lm, i2w, tokens, blank, device, box, args.beam, args.source_id, args.folio)
        row["bbox"] = box
        rows.append(row)
    print(json.dumps({"image": str(args.image), "regions": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
