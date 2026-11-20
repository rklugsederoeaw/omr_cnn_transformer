"""Lesezeit des CRNN je System, Einzelprozess, Batch 1, auf der GPU des Jobs (Artikel, Abschnitt 6.4).

    MODELL=hybrid_hybrid_m4g_s0 LM=decoder_lm DATA=<Index-Ordner> $TORCH -I skripte/lesezeit_crnn.py

Gemessen je System nach 10 Aufwärmläufen, mit torch.cuda.synchronize: (a) Encoder und CTC-Greedy-Lesung, (b) gemeinsame
Decodierung wie joint.run (Beam 6, CTC- und Attention-Bewertung jedes Kandidaten, Melodiemodell als dritte Stimme).
Die volle Lesekette (Beam 12, Schlüsselvarianten, Konkordanzen) ist nicht enthalten. Bildladen zählt nicht mit.
Ausgabe: Mittel, Median und 90-%-Quantil in ms je System auf stdout.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(
    0, os.environ.get("OMR_CLI", "/home/rklug/projects_2026/omr_cli") + "/src"
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MODUS", "hybrid")
import hybrid as H  # noqa: E402
import joint as J  # noqa: E402
from core.data.datamodule import pad_collate  # noqa: E402
from core.data.dataset import CTCDataset  # noqa: E402
from core.models.crnn import CRNN_OMR  # noqa: E402
from decoder_lm import MelodyLM  # noqa: E402


@torch.no_grad()
def main():
    dev = "cuda"
    i2w = {
        int(k): v
        for k, v in json.load((H.CLI / "vocab/volpiano3_musici2w.json").open()).items()
    }
    blank = len(i2w)
    lmck = torch.load(H.HERE / "modelle/decoder_lm.pt", map_location="cpu")
    voc = lmck["vocab"]
    w2i = {w: i for i, w in enumerate(voc)}
    model = H.Hybrid(CRNN_OMR(blank + 1), None, blank + 1, len(voc))
    if not model.use_dec:
        H.MODUS = "scratch"
        model = H.Hybrid(CRNN_OMR(blank + 1), None, blank + 1, len(voc))
    model.load_state_dict(
        torch.load(
            H.HERE / "modelle" / f"{os.environ['MODELL']}.pt", map_location="cpu"
        )
    )
    model.to(dev).eval()
    lck = torch.load(H.HERE / "modelle" / f"{os.environ['LM']}.pt", map_location="cpu")
    lm = MelodyLM(len(lck["vocab"]), **lck["cfg"]).to(dev).eval()
    lm.load_state_dict(lck["state"])
    ds = CTCDataset(H.DATA, "test", "music", augment=False)
    ds.set_dictionaries({v: k for k, v in i2w.items()}, i2w)
    stuecke = [pad_collate([ds[k]]) for k in range(len(ds))]  # Bildladen vorab

    def ctc(b):
        X, _, FL, _ = b
        h = model.encode(X.to(dev))
        H.ctc_greedy(model.ctc(h).log_softmax(-1)[0], FL[0].item(), blank, i2w)

    def gemeinsam(b):
        J.run(model, [b], i2w, w2i, voc, blank, dev, lm=lm)

    for name, f in (
        ("CTC greedy", ctc),
        ("gemeinsam, Beam 6, mit Melodiemodell", gemeinsam),
    ):
        for b in stuecke[:10]:
            f(b)
        t = []
        for b in stuecke:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            f(b)
            torch.cuda.synchronize()
            t.append(1000 * (time.perf_counter() - t0))
        t.sort()
        print(
            f"{name}: {len(t)} Systeme, Mittel {statistics.mean(t):.1f} ms, Median {statistics.median(t):.1f} ms, "
            f"90 % {t[int(0.9 * len(t))]:.1f} ms",
            flush=True,
        )
    print(
        torch.cuda.get_device_name(0),
        "| Parameter",
        sum(p.numel() for p in model.parameters()),
    )


if __name__ == "__main__":
    main()
