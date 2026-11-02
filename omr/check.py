"""Check that the bundled weights still read the reference notation line."""
import json
from pathlib import Path

import torch

from omr.predict import ROOT, load_models, predict


def main():
    torch.set_num_threads(4)
    device = torch.device("cpu")
    image = ROOT / "examples/001r_01_with_clef.jpg"
    expected = json.loads((ROOT / "examples/001r_01_expected.json").read_text())
    models = load_models(device, ROOT / "weights/hybrid_hybrid_general_v10.pt", ROOT / "weights/decoder_lm_diff.pt")
    actual = predict(image, *models, device, source_id=123667, folio="001r")
    if actual["gelesen"] != expected:
        raise SystemExit("Model check failed: reading differs from the reference")
    print(f"OK: {len(expected)} tokens exactly matched")


if __name__ == "__main__":
    main()
