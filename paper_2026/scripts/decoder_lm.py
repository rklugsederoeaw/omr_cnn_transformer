"""Schritt 2: Melodie-Decoder (Transformer, nächstes Token) auf dem Korpus daten/decoder_korpus/korpus.jsonl vortrainieren.

Decoder-only mit kausaler Maske; später kommen Cross-Attention-Schichten auf die CRNN-Encoderausgabe dazu (Schritt 3).
Vokabular = 65 Tokens der CRNN-Ground-Truth + <pad> <bos> <eos>. Validierung: 5 % der Cantus-IDs (Feld split).
Aufruf: /home/rklug/projects_2026/omr_cli/.venv/bin/python -I skripte/decoder_lm.py [EPOCHEN]
Ausgabe: modelle/decoder_lm.pt (bester Stand), modelle/decoder_lm_vocab.json, docs/decoder_lm.md, daten/decoder_lm.done
"""

import json
import os
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
EPOCHEN = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12
D, LAYERS, HEADS, FF, MAXLEN, BATCH, LR = 256, 6, 4, 1024, 224, 256, 6e-4
KORPUS = os.environ.get("KORPUS", "decoder_korpus")  # Verzeichnis unter daten/
NAME = os.environ.get("NAME", "decoder_lm")  # Ausgabename unter modelle/ und docs/
torch.manual_seed(0)


class MelodyLM(nn.Module):
    def __init__(self, v, d=D, layers=LAYERS, heads=HEADS, ff=FF, maxlen=MAXLEN):
        super().__init__()
        self.tok = nn.Embedding(v, d)
        self.pos = nn.Embedding(maxlen, d)
        layer = nn.TransformerEncoderLayer(
            d, heads, ff, dropout=0.1, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, v, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, x):
        n = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=x.device)
        return self.head(self.norm(self.body(h, mask=mask, is_causal=True)))


def load(vocab):
    w2i = {w: i for i, w in enumerate(vocab)}
    tr, va = [], []
    for line in (HERE / "daten" / KORPUS / "korpus.jsonl").open():
        r = json.loads(line)
        ids = (
            [w2i["<bos>"]]
            + [w2i[t] for t in r["tokens"].split()][: MAXLEN - 2]
            + [w2i["<eos>"]]
        )
        (va if r["split"] == "val" else tr).append(ids)
    return tr, va


def batches(data, shuffle, dev):
    idx = torch.randperm(len(data)).tolist() if shuffle else list(range(len(data)))
    for i in range(0, len(idx), BATCH):
        seqs = [data[j] for j in idx[i : i + BATCH]]
        n = max(map(len, seqs))
        x = torch.zeros(len(seqs), n, dtype=torch.long)
        for k, s in enumerate(seqs):
            x[k, : len(s)] = torch.tensor(s)
        yield x.to(dev)


def evaluate(model, data, dev):
    model.eval()
    loss_sum = n_tok = correct = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x in batches(data, False, dev):
            logits = model(x[:, :-1]).float()
            y = x[:, 1:]
            m = y != 0
            loss_sum += nn.functional.cross_entropy(
                logits[m], y[m], reduction="sum"
            ).item()
            correct += (logits.argmax(-1)[m] == y[m]).sum().item()
            n_tok += m.sum().item()
    model.train()
    return loss_sum / n_tok, correct / n_tok


def main():
    dev = "cuda"
    vocab = ["<pad>", "<bos>", "<eos>"] + json.load(
        (ROOT / "trainingsdaten" / "vokabular.json").open()
    )
    tr, va = load(vocab)
    model = MelodyLM(len(vocab)).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    steps = EPOCHEN * math.ceil(len(tr) / BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, LR, total_steps=steps, pct_start=0.05
    )
    (HERE / "modelle").mkdir(exist_ok=True)
    json.dump(vocab, (HERE / "modelle" / f"{NAME}_vocab.json").open("w"))
    rows, best, t0 = [], 9e9, time.time()
    # Bezug: Unigramm-Perplexität auf Validierung
    from collections import Counter

    cnt = Counter(t for s in tr for t in s[1:])
    tot = sum(cnt.values())
    uni = -sum(math.log(cnt[t] / tot) for s in va for t in s[1:]) / sum(
        len(s) - 1 for s in va
    )
    for ep in range(1, EPOCHEN + 1):
        for x in batches(tr, True, dev):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x[:, :-1])
            y = x[:, 1:]
            loss = nn.functional.cross_entropy(
                logits.float().reshape(-1, len(vocab)), y.reshape(-1), ignore_index=0
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vl, vacc = evaluate(model, va, dev)
        rows.append((ep, loss.item(), vl, math.exp(vl), vacc, time.time() - t0))
        print(
            f"Epoche {ep}: Train {loss.item():.3f}  Val {vl:.3f}  Perplexität {math.exp(vl):.2f}  Treffer nächstes Token {vacc:.1%}  {time.time() - t0:.0f} s",
            flush=True,
        )
        if vl < best:
            best = vl
            torch.save(
                {
                    "state": model.state_dict(),
                    "vocab": vocab,
                    "cfg": dict(d=D, layers=LAYERS, heads=HEADS, ff=FF, maxlen=MAXLEN),
                },
                HERE / "modelle" / f"{NAME}.pt",
            )
    lines = [
        "# Melodie-Decoder: Vortraining",
        "",
        f"`skripte/decoder_lm.py`: Transformer-Decoder (kausal), {LAYERS} Schichten, d = {D}, {HEADS} Köpfe, {n_par / 1e6:.1f} Mio. Parameter, Vokabular {len(vocab)}. Training {len(tr)} Sequenzen, Validierung {len(va)} (5 % der Cantus-IDs, alle Fassungen gemeinsam). Batch {BATCH}, AdamW, OneCycle bis {LR}, bfloat16, RTX 5090. Bezug: Unigramm-Perplexität {math.exp(uni):.2f}.",
        "",
        "| Epoche | Train-Verlust | Val-Verlust | Val-Perplexität | Treffer nächstes Token | Zeit (s) |",
        "|---|---|---|---|---|---|",
    ] + [
        f"| {e} | {a:.3f} | {b:.3f} | {c:.2f} | {d:.1%} | {t:.0f} |"
        for e, a, b, c, d, t in rows
    ]
    (HERE / "docs" / f"{NAME}.md").write_text("\n".join(lines) + "\n")
    (HERE / "daten" / f"{NAME}.done").write_text("ok\n")


if __name__ == "__main__":
    m = MelodyLM(10, d=16, layers=1, heads=2, ff=32, maxlen=8)
    assert m(torch.zeros(2, 5, dtype=torch.long)).shape == (2, 5, 10)
    main()
