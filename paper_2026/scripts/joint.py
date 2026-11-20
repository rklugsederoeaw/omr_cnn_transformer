"""Gemeinsame Decodierung eines Hybrids: Attention-Beam-Search (BEAM Kandidaten) plus CTC-Greedy-Lesung als Kandidat;
jeder Kandidat wird bewertet mit lam * log P_CTC(y|x) + (1 - lam) * log P_Att(y|x) (+ beta * Länge). Gewichte auf der
Validierung gewählt, dann auf der festen Testmenge gemessen.
Aufruf: MODELL=hybrid_hybrid_lang /home/rklug/projects_2026/omr_cli/.venv/bin/python -I skripte/joint.py
Ausgabe: an docs/hybrid_<modell>.md angehängt."""

import json
import os
import sys
from itertools import groupby
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(
    0, os.environ.get("OMR_CLI", "/home/rklug/projects_2026/omr_cli") + "/src"
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("MODUS", "hybrid")
import hybrid as H  # noqa: E402
from core.data.datamodule import pad_collate  # noqa: E402
from core.data.dataset import CTCDataset  # noqa: E402
from core.models.crnn import CRNN_OMR  # noqa: E402
from decoder_lm import MelodyLM  # noqa: E402

HERE = H.HERE
MODELL = os.environ.get("MODELL", "hybrid_hybrid_lang")
BEAM = int(os.environ.get("BEAM", "6"))
LAMS = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
LM = os.environ.get(
    "LM", ""
)  # dritte Stimme: unverändertes Melodiemodell unter modelle/<LM>.pt (Shallow Fusion)
MUS = [0.0, 0.05, 0.1, 0.2, 0.3] if LM else [0.0]
BETAS = [0.0, 0.5, 1.0]


@torch.no_grad()
def beam_search(model, h, fl, bos, eos, beam):
    """Je Zeile die BEAM besten Attention-Hypothesen (Tokenlisten ohne bos/eos) mit Summe der Log-Wahrsch."""
    B = h.shape[0]
    mem = model.memory(h)
    pad = torch.arange(h.shape[1], device=h.device)[None] >= fl[:, None]
    mem = mem.repeat_interleave(beam, 0)
    pad = pad.repeat_interleave(beam, 0)
    y = torch.full((B * beam, 1), bos, device=h.device)
    score = torch.full((B, beam), -1e9, device=h.device)
    score[:, 0] = 0
    score = score.view(-1)
    fin = torch.zeros(B * beam, dtype=torch.bool, device=h.device)
    done = [[] for _ in range(B)]
    for _ in range(H.MAXOUT):
        lp = model.decode(y, mem, pad)[:, -1].float().log_softmax(-1)
        V = lp.shape[-1]
        lp[fin] = -1e9
        cand = (score[:, None] + lp).view(B, beam * V)
        top, idx = cand.topk(beam, -1)
        src = (idx // V) + torch.arange(B, device=h.device)[:, None] * beam
        tok = idx % V
        y = torch.cat([y[src.view(-1)], tok.view(-1, 1)], 1)
        score = top.view(-1)
        is_eos = tok.view(-1) == eos
        for k in torch.nonzero(is_eos).view(-1).tolist():
            done[k // beam].append((y[k, 1:-1].tolist(), score[k].item()))
        score = score.masked_fill(is_eos, -1e9)
        fin = score < -1e8
        if all(len(d) >= beam for d in done) or fin.all():
            break
    return [sorted(d, key=lambda x: -x[1])[:beam] for d in done]


@torch.no_grad()
def att_score(model, h, fl, seqs, bos, eos):
    """log P_Att(seq) je Kandidat (eine Zeile, mehrere Kandidaten)."""
    n = max(map(len, seqs)) + 2
    x = torch.zeros(len(seqs), n, dtype=torch.long, device=h.device)
    for k, s in enumerate(seqs):
        x[k, : len(s) + 2] = torch.tensor([bos] + s + [eos])
    mem = model.memory(h).expand(len(seqs), -1, -1)
    pad = (torch.arange(h.shape[1], device=h.device)[None] >= fl).expand(len(seqs), -1)
    lp = model.decode(x[:, :-1], mem, pad).float().log_softmax(-1)
    tgt = x[:, 1:]
    s = lp.gather(-1, tgt[..., None])[..., 0] * (tgt != 0)
    return s.sum(1)


@torch.no_grad()
def lm_score(lm, seqs, bos, eos, dev):
    if lm is None:
        return [0.0] * len(seqs)
    # ponytail: Das Melodiemodell kennt nur pos.num_embeddings Positionen (224); längere Kandidaten (Beam bis 230,
    # CTC-Lesung unbegrenzt) werden auf ihre ersten Token gekürzt bewertet (bos + Token + eos passen dann). Vorher
    # brach die Bewertung mit einem CUDA-Indexfehler ab (MUSICA-Job 1954701, hybrid_scratch_m1_s1).
    seqs = [s[: lm.pos.num_embeddings - 2] for s in seqs]
    n = max(map(len, seqs)) + 2
    x = torch.zeros(len(seqs), n, dtype=torch.long, device=dev)
    for k, s in enumerate(seqs):
        x[k, : len(s) + 2] = torch.tensor([bos] + s + [eos])
    lp = lm(x[:, :-1]).float().log_softmax(-1)
    tgt = x[:, 1:]
    return (lp.gather(-1, tgt[..., None])[..., 0] * (tgt != 0)).sum(1).tolist()


def run(model, loader, i2w_ctc, w2i_dec, voc, blank, dev, limit=None, lm=None):
    """Kandidaten und Teilwerte je Zeile sammeln, damit die Gewichte ohne Neurechnen gewählt werden können."""
    bos, eos = w2i_dec["<bos>"], w2i_dec["<eos>"]
    dec2ctc = {w2i_dec[v]: k for k, v in i2w_ctc.items()}
    rows = []
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for X, Y, FL, TL in loader:
            X, FL = X.to(dev), FL.to(dev)
            h = model.encode(X)
            lpc = model.ctc(h).float().log_softmax(-1)
            beams = beam_search(model, h, FL, bos, eos, BEAM)
            for k in range(X.shape[0]):
                gt = [i2w_ctc[t] for t in Y[k, : TL[k]].tolist()]
                g = [
                    c
                    for c, _ in groupby(lpc[k, : FL[k]].argmax(-1).tolist())
                    if c != blank
                ]
                cands = [s for s, _ in beams[k] if all(t in dec2ctc for t in s)]
                gd = [w2i_dec[i2w_ctc[c]] for c in g]
                if gd not in cands:
                    cands.append(gd)
                cands = [c for c in cands if c] or [gd]
                a = att_score(model, h[k : k + 1], FL[k], cands, bos, eos).tolist()
                tgt = [torch.tensor([dec2ctc[t] for t in c] or [0]) for c in cands]
                lp_rep = (
                    lpc[k : k + 1, : FL[k]].permute(1, 0, 2).expand(-1, len(cands), -1)
                )
                c_ll = -nn.functional.ctc_loss(
                    lp_rep,
                    nn.utils.rnn.pad_sequence(tgt, batch_first=True).to(dev),
                    FL[k].repeat(len(cands)),
                    torch.tensor([len(t) for t in tgt], device=dev),
                    blank=blank,
                    reduction="none",
                    zero_infinity=True,
                )
                rows.append(
                    (
                        gt,
                        [[voc[t] for t in c] for c in cands],
                        a,
                        c_ll.tolist(),
                        lm_score(lm, cands, bos, eos, dev),
                    )
                )
            if limit and len(rows) >= limit:
                break
    return rows


def pick(cands, a, c, m, lam, beta, mu):
    return max(
        range(len(cands)),
        key=lambda i: lam * c[i] + (1 - lam) * a[i] + mu * m[i] + beta * len(cands[i]),
    )


def ser(rows, lam, beta, mu=0.0):
    ed = L = 0
    for gt, cands, a, c, m in rows:
        best = pick(cands, a, c, m, lam, beta, mu)
        ed += H.lev(cands[best], gt)
        L += len(gt)
    return 100 * ed / L


def main():
    dev = "cuda"
    i2w_raw = json.load((H.CLI / "vocab" / "volpiano3_musici2w.json").open())
    i2w_ctc = {int(k): v for k, v in i2w_raw.items()}
    for t in H.VOK_EXTRA:  # erweitertes Ausgabevokabular (general_v8: c5)
        i2w_ctc[len(i2w_ctc)] = t
    blank = len(i2w_ctc)
    lmck = torch.load(HERE / "modelle" / "decoder_lm.pt", map_location="cpu")
    voc = lmck["vocab"] + [t for t in H.VOK_EXTRA if t not in lmck["vocab"]]
    w2i_dec = {w: i for i, w in enumerate(voc)}
    model = H.Hybrid(CRNN_OMR(blank + 1), None, blank + 1, len(voc))
    if (
        not model.use_dec
    ):  # Hybrid ohne lm baut den Decoder nur im Modus scratch; hier erzwingen
        H.MODUS = "scratch"
        model = H.Hybrid(CRNN_OMR(blank + 1), None, blank + 1, len(voc))
    model.load_state_dict(
        torch.load(HERE / "modelle" / f"{MODELL}.pt", map_location="cpu")
    )
    model.to(dev)
    lmmod = None
    if LM:
        lck = torch.load(HERE / "modelle" / f"{LM}.pt", map_location="cpu")
        lmmod = MelodyLM(len(lck["vocab"]), **lck["cfg"]).to(dev).eval()
        lmmod.load_state_dict(lck["state"])
        assert lck["vocab"] == voc
    out = {}
    for split in ("val", "test"):
        ds = CTCDataset(H.DATA, split, "music", augment=False)
        ds.set_dictionaries({v: k for k, v in i2w_ctc.items()}, i2w_ctc)
        if split == "val":
            ds = torch.utils.data.Subset(ds, list(range(0, len(ds), 3)))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=16, shuffle=False, num_workers=6, collate_fn=pad_collate
        )
        out[split] = run(model, loader, i2w_ctc, w2i_dec, voc, blank, dev, lm=lmmod)
        print(split, len(out[split]), "Zeilen", flush=True)
    grid = {(lam, beta): ser(out["val"], lam, beta) for lam in LAMS for beta in BETAS}
    lam0, beta0 = min(grid, key=grid.get)
    mgrid = {mu: ser(out["val"], lam0, beta0, mu) for mu in MUS}
    mu = min(mgrid, key=mgrid.get)
    lam, beta = min(grid, key=grid.get)
    lines = [
        "",
        f"## Gemeinsame Decodierung (`skripte/joint.py`, Beam {BEAM})",
        "",
        f"Kandidaten: {BEAM} Attention-Hypothesen aus der Beam-Search plus die CTC-Greedy-Lesung. Bewertung lam · log P_CTC + (1 − lam) · log P_Att + beta · Länge; lam und beta auf {len(out['val'])} Validierungszeilen gewählt: lam = {lam}, beta = {beta}.",
        "",
        "| Verfahren | Val-SER | Test-SER |",
        "|---|---|---|",
    ]
    for name, (l_, b_) in [
        ("nur Attention (bester Beam-Kandidat)", (0.0, 0.0)),
        ("nur CTC (beste CTC-Bewertung unter den Kandidaten)", (1.0, 0.0)),
        ("gemeinsam", (lam, beta)),
    ]:
        lines.append(
            f"| {name} | {ser(out['val'], l_, b_):.2f} | {ser(out['test'], l_, b_):.2f} |"
        )
    lines += [
        "",
        "Val-SER je lam (beta = gewählt): "
        + ", ".join(f"{lm_}: {grid[(lm_, beta)]:.2f}" for lm_ in LAMS)
        + ".",
    ]
    if LM:
        changed = better = worse = 0
        for gt, cands, a, c, m in out["test"]:
            i0, i1 = (
                pick(cands, a, c, m, lam0, beta0, 0.0),
                pick(cands, a, c, m, lam0, beta0, mu),
            )
            if i0 != i1:
                changed += 1
                d = H.lev(cands[i1], gt) - H.lev(cands[i0], gt)
                better += d < 0
                worse += d > 0
        lines += [
            "",
            f"Dritte Stimme `{LM}` (Shallow Fusion), Gewicht mu auf der Validierung gewählt: mu = {mu}. Val-SER je mu: "
            + ", ".join(f"{k}: {v:.2f}" for k, v in mgrid.items())
            + ".",
            "",
            "| Verfahren | Val-SER | Test-SER |",
            "|---|---|---|",
            f"| gemeinsam ohne Melodiemodell | {ser(out['val'], lam0, beta0):.2f} | {ser(out['test'], lam0, beta0):.2f} |",
            f"| gemeinsam mit Melodiemodell | {ser(out['val'], lam0, beta0, mu):.2f} | {ser(out['test'], lam0, beta0, mu):.2f} |",
            "",
            f"Testzeilen, deren Lesung das Melodiemodell ändert: {changed}; davon besser {better}, schlechter {worse}, gleich gut {changed - better - worse}.",
        ]
    with (HERE / "docs" / f"{MODELL}.md").open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
