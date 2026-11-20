"""Schritt 3: Hybrid aus CRNN-Encoder (v3) und vortrainiertem Melodie-Decoder, CTC-Zweig bleibt.

Encoder: Faltungsteil des CRNN v3; BiLSTM mit batch_first=True (im CLI läuft die Rekurrenz über die Batch-Dimension,
siehe docs/hybrid.md), mit den v3-Gewichten initialisiert; CTC-Kopf aus v3. Decoder: Transformer mit Self-Attention und
Feedforward aus modelle/decoder_lm.pt, neue Cross-Attention auf die BiLSTM-Ausgabe. Verlust 0,3 CTC + 0,7 Kreuzentropie.
MODUS: hybrid (Decoder vortrainiert), scratch (Decoder zufällig), ctc (nur CTC, korrigiertes LSTM).
Aufruf: MODUS=hybrid EPOCHEN=15 /home/rklug/projects_2026/omr_cli/.venv/bin/python -I skripte/hybrid.py
Ausgabe: modelle/hybrid_<modus>.pt, docs/hybrid_<modus>.md, daten/hybrid_<modus>.done
"""

import json

import cv2
import os
import random
import sys
import time
from itertools import groupby
from pathlib import Path

import torch
import torch.nn as nn

cv2.setNumThreads(1)  # Datenlader teilen sich die CPU mit anderen Läufen
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

CLI = Path(
    os.environ.get("OMR_CLI", "/home/rklug/projects_2026/omr_cli")
)  # Cluster: eigene Ablage
sys.path.insert(0, str(CLI / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data.datamodule import pad_collate  # noqa: E402
from core.data.dataset import CTCDataset  # noqa: E402
from core.models.crnn import CRNN_OMR  # noqa: E402
from decoder_lm import MelodyLM  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
DATA = os.environ.get(
    "DATA", "/home/rklug/projects_2026/volpiano2omr_gt/trainingsdaten/omr_gt"
)
# VOK_EXTRA: zusätzliche Ausgabetokens, Komma getrennt (mb2: c5 auf Fünfliniensystemen). Die Gewichte des
# Startmodells werden um je eine Zeile erweitert, als Vorbild dient VOK_VORBILD (Standard c4).
VOK_EXTRA = [t for t in os.environ.get("VOK_EXTRA", "").split(",") if t]
VOK_VORBILD = os.environ.get("VOK_VORBILD", "c4")
# VOK_START: "vorbild" (Zeile des Vorbildtokens mit kleiner Störung) oder "zufall" (frisch gezogen wie ein
# neues Token). Mit "vorbild" blieb c5 dem c4 zu ähnlich, der Decoder trennte beide nicht (mb2, 23.10.2026).
VOK_START = os.environ.get("VOK_START", "vorbild")
INIT = os.environ.get(
    "INIT", "v3"
)  # v3: Encoder aus CRNN v3; none: zufällig (z. B. für Leave-one-source-out); hybrid_<name>: ganzes Modell aus modelle/ (Feintuning)
ENC_LR = float(os.environ.get("ENC_LR", "2e-4"))
MODUS = os.environ.get("MODUS", "hybrid")
NAME = MODUS + os.environ.get("TAG", "")
NUR_TEST = (
    os.environ.get("NUR_TEST") == "1"
)  # nur gespeicherten Stand auf der Testmenge messen
FREEZE = int(
    os.environ.get("FREEZE", "0")
)  # Epochen mit eingefrorenen vortrainierten Decoder-Teilen
LR_PRE = float(
    os.environ.get("LR_PRE", "4e-4")
)  # Spitzen-Lernrate der vortrainierten Decoder-Teile  # Ausgabename, z. B. hybrid_lang
EPOCHEN = int(os.environ.get("EPOCHEN", "15"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))  # > 0: Probelauf
BATCH, WORKERS, LAMBDA_CTC, MAXOUT = 16, int(os.environ.get("WORKERS", "10")), 0.3, 230
VAL_N = 800
# SEED: Initialisierung und Reihenfolge der Batches; die Auswahl der Validierungszeilen bleibt fest (Random(1))
SEED = int(os.environ.get("SEED", "0"))
torch.manual_seed(SEED)
random.seed(SEED)
# LM_INIT: Melodiemodell für Vokabular und Decoderstart (Leave-one-source-out: decoder_lm_ohne<id>)
LM_INIT = os.environ.get("LM_INIT", "decoder_lm")


class Hybrid(nn.Module):
    def __init__(
        self,
        crnn: CRNN_OMR,
        lm: MelodyLM | None,
        n_ctc: int,
        n_dec: int,
        d=256,
        heads=4,
        ff=1024,
        layers=6,
    ):
        super().__init__()
        self.backbone = crnn.backbone
        self.rnn_features = crnn.rnn_features
        self.rnn = nn.LSTM(
            self.rnn_features, 256, num_layers=2, bidirectional=True, batch_first=True
        )
        self.rnn.load_state_dict(
            crnn.rnn[0].state_dict()
        )  # gleiche Gewichtsform, jetzt über die Zeit
        self.ctc = nn.Linear(512, n_ctc)
        self.ctc.load_state_dict(crnn.vocab_projection.state_dict())
        self.use_dec = lm is not None or MODUS == "scratch"
        if not self.use_dec:
            return
        self.mem = nn.Linear(512, d)
        self.mem_pos = nn.Embedding(4096, d)
        self.tok = nn.Embedding(n_dec, d)
        self.pos = nn.Embedding(256, d)
        layer = nn.TransformerDecoderLayer(
            d, heads, ff, dropout=0.1, batch_first=True, norm_first=True
        )
        self.dec = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_dec, bias=False)
        if lm is not None:
            self.tok.load_state_dict(lm.tok.state_dict())
            self.pos.weight.data[: lm.pos.weight.shape[0]] = lm.pos.weight.data
            self.norm.load_state_dict(lm.norm.state_dict())
            for dl, el in zip(self.dec.layers, lm.body.layers):
                dl.self_attn.load_state_dict(el.self_attn.state_dict())
                dl.linear1.load_state_dict(el.linear1.state_dict())
                dl.linear2.load_state_dict(el.linear2.state_dict())
                dl.norm1.load_state_dict(el.norm1.state_dict())
                dl.norm3.load_state_dict(el.norm2.state_dict())
        self.head.weight = self.tok.weight

    def encode(self, x):
        h = self.backbone(x)
        h = h.permute(0, 3, 1, 2).reshape(h.shape[0], h.shape[3], self.rnn_features)
        h, _ = self.rnn(h)
        return h  # (B, T, 512)

    def memory(self, h):
        return self.mem(h) + self.mem_pos(torch.arange(h.shape[1], device=h.device))

    def decode(self, y_in, mem, mem_pad):
        n = y_in.shape[1]
        t = self.tok(y_in) + self.pos(torch.arange(n, device=y_in.device))
        mask = nn.Transformer.generate_square_subsequent_mask(n, device=y_in.device)
        return self.head(
            self.norm(
                self.dec(
                    t,
                    mem,
                    tgt_mask=mask,
                    tgt_is_causal=True,
                    memory_key_padding_mask=mem_pad,
                )
            )
        )


def lev(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


def ctc_greedy(lp, n, blank, i2w):
    best = [k for k, _ in groupby(lp[:n].argmax(-1).tolist())]
    return [i2w[c] for c in best if c != blank]


@torch.no_grad()
def att_greedy(model, h, feat_len, voc, bos, eos):
    mem = model.memory(h)
    pad = torch.arange(h.shape[1], device=h.device)[None] >= feat_len[:, None]
    y = torch.full((h.shape[0], 1), bos, device=h.device)
    done = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
    for _ in range(MAXOUT):
        nxt = model.decode(y, mem, pad)[:, -1].argmax(-1)
        y = torch.cat([y, nxt[:, None]], 1)
        done |= nxt == eos
        if done.all():
            break
    out = []
    for row in y[:, 1:].tolist():
        out.append([voc[t] for t in (row[: row.index(eos)] if eos in row else row)])
    return out


def evaluate(model, loader, i2w_ctc, blank, voc, bos, eos, dev, limit=None):
    model.eval()
    ed_c = ed_a = L = n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for X, Y, FL, TL in loader:
            X, FL = X.to(dev), FL.to(dev)
            h = model.encode(X)
            lp = model.ctc(h).float().log_softmax(-1)
            att = att_greedy(model, h, FL, voc, bos, eos) if model.use_dec else None
            for k in range(X.shape[0]):
                gt = [i2w_ctc[t] for t in Y[k, : TL[k]].tolist()]
                ed_c += lev(ctc_greedy(lp[k], FL[k].item(), blank, i2w_ctc), gt)
                if att is not None:
                    ed_a += lev(att[k], gt)
                L += len(gt)
            n += X.shape[0]
            if limit and n >= limit:
                break
    model.train()
    return 100 * ed_c / L, (100 * ed_a / L if model.use_dec else None)


def vokabular_erweitern(sd, k, vorbild, voc):
    """Gewichte eines Startmodells um k neue Tokens erweitern (VOK_EXTRA).

    CTC-Kopf: die neuen Zeilen stehen vor der Leerzeichenklasse (blank bleibt die letzte), Decoder-Embedding
    (mit dem Ausgabekopf geteilt) wächst am Ende. Startwerte sind die Zeilen des Vorbildtokens."""
    ctc_w, ctc_b = sd["ctc.weight"], sd["ctc.bias"]
    n = ctc_w.shape[0] - 1  # Tokens ohne blank
    g = torch.Generator().manual_seed(2)
    neue = neue_zeilen(ctc_w, k, vorbild, g)
    sd["ctc.weight"] = torch.cat([ctc_w[:n], neue, ctc_w[n:]])
    sd["ctc.bias"] = torch.cat(
        [ctc_b[:n], ctc_b[vorbild : vorbild + 1].repeat(k), ctc_b[n:]]
    )
    return zeilen_anhaengen(
        sd, ("tok.weight", "head.weight"), k, voc.index(VOK_VORBILD)
    )


def zeilen_anhaengen(sd, keys, k, vb):
    """Je Gewichtsmatrix k Zeilen anhängen, Startwert ist Zeile vb (Vorbildtoken) mit kleiner Störung.

    Eine exakte Kopie macht altes und neues Token symmetrisch. Im Decoder sind Embedding und Ausgabekopf
    dieselben Gewichte, die Symmetrie bleibt dann im Training lange erhalten, und argmax nimmt stets den
    kleineren Index: der Attention-Zweig gab c5 nie aus (mb2, 23.10.2026). Die Störung bricht sie."""
    g = torch.Generator().manual_seed(1)
    for key in keys:
        if key in sd:
            w = sd[key]
            sd[key] = torch.cat([w, neue_zeilen(w, k, vb, g)])
    return sd


def neue_zeilen(w, k, vb, g):
    """k Startzeilen für neue Tokens: Vorbildzeile mit Störung oder frei gezogen (VOK_START)."""
    rausch = torch.randn((k, w.shape[1]), generator=g).to(w.dtype).to(w.device)
    if VOK_START == "zufall":
        return w.std() * rausch
    return w[vb : vb + 1].repeat(k, 1) + 0.1 * w.std() * rausch


def main():
    dev = "cuda"
    i2w_raw = json.load((CLI / "vocab" / "volpiano3_musici2w.json").open())
    i2w_ctc = {int(k): v for k, v in i2w_raw.items()}
    for t in VOK_EXTRA:
        i2w_ctc[len(i2w_ctc)] = t
    w2i_ctc = {v: k for k, v in i2w_ctc.items()}
    blank = len(i2w_ctc)
    ck = torch.load(
        CLI / "weights" / "music" / "CRNN_music_volpiano3.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    crnn = CRNN_OMR(blank + 1)
    if INIT == "v3":
        crnn.load_state_dict(
            {
                k.removeprefix("model."): v
                for k, v in ck["state_dict"].items()
                if k.startswith("model.")
            }
        )
    lmck = torch.load(HERE / "modelle" / f"{LM_INIT}.pt", map_location="cpu")
    voc = lmck["vocab"] + [t for t in VOK_EXTRA if t not in lmck["vocab"]]
    w2i_dec = {w: i for i, w in enumerate(voc)}
    lm = None
    if MODUS == "hybrid":
        lm = MelodyLM(len(voc), **lmck["cfg"])
        st = lmck["state"]
        if VOK_EXTRA:  # Melodiemodell kennt die neuen Tokens nicht
            st = zeilen_anhaengen(
                st,
                ("tok.weight", "head.weight"),
                len(VOK_EXTRA),
                lmck["vocab"].index(VOK_VORBILD),
            )
        lm.load_state_dict(st)
    model = Hybrid(crnn, lm if MODUS != "ctc" else None, blank + 1, len(voc)).to(dev)
    if MODUS == "ctc":
        model.use_dec = False
    if INIT.startswith("hybrid_"):
        sd = torch.load(HERE / "modelle" / f"{INIT}.pt", map_location=dev)
        if VOK_EXTRA:
            sd = vokabular_erweitern(sd, len(VOK_EXTRA), w2i_ctc[VOK_VORBILD], voc)
        model.load_state_dict(sd)
    ctc_to_dec = torch.tensor([w2i_dec[i2w_ctc[i]] for i in range(blank)])
    sets = {}
    for split, aug in [("train", True), ("val", False), ("test", False)]:
        ds = CTCDataset(DATA, split, "music", augment=aug)
        ds.set_dictionaries(w2i_ctc, i2w_ctc)
        sets[split] = ds
    val_idx = random.Random(1).sample(
        range(len(sets["val"])), min(VAL_N, len(sets["val"]))
    )

    def mk(ds, sh):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=BATCH,
            shuffle=sh,
            num_workers=WORKERS,
            collate_fn=pad_collate,
            persistent_workers=True,
        )

    tr_loader = mk(sets["train"], True)
    va_loader = mk(torch.utils.data.Subset(sets["val"], val_idx), False)
    te_loader = mk(sets["test"], False)
    enc_params = (
        list(model.backbone.parameters())
        + list(model.rnn.parameters())
        + list(model.ctc.parameters())
    )
    vor = [  # vortrainierte Decoder-Teile (nur MODUS=hybrid sinnvoll)
        p
        for n_, p in model.named_parameters()
        if n_.startswith(("tok.", "pos.", "norm."))
        or (
            n_.startswith("dec.layers.")
            and any(
                s in n_
                for s in (".self_attn.", ".linear1.", ".linear2.", ".norm1.", ".norm3.")
            )
        )
    ]
    rest = [p for p in model.parameters() if not any(p is q for q in enc_params + vor)]
    lrs = [ENC_LR, LR_PRE, float(os.environ.get("LR_REST", "4e-4"))]
    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lrs[0]},
            {"params": vor, "lr": lrs[1]},
            {"params": rest, "lr": lrs[2]},
        ],
        weight_decay=0.01,
    )
    steps = MAX_STEPS or EPOCHEN * len(tr_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, lrs, total_steps=steps, pct_start=0.05
    )

    def set_frozen(flag):
        for p in vor:
            p.requires_grad_(not flag)

    set_frozen(FREEZE > 0)
    ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)
    bos, eos = w2i_dec["<bos>"], w2i_dec["<eos>"]
    rows, best, step, t0, ep0 = [], 9e9, 0, time.time(), 1
    stand = HERE / "modelle" / f"hybrid_{NAME}_stand.pt"
    if stand.exists() and not NUR_TEST:  # nach Abbruch fortsetzen
        s = torch.load(stand, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        rows, best, step, ep0, base_c, base_a = (
            s["rows"],
            s["best"],
            s["step"],
            s["ep"] + 1,
            s["base_c"],
            s["base_a"],
        )
        print(f"fortgesetzt nach Epoche {s['ep']}", flush=True)
    else:
        base_c, base_a = evaluate(model, va_loader, i2w_ctc, blank, voc, bos, eos, dev)
    print(
        f"vor dem Training: Val-SER CTC {base_c:.2f} %  Attention {base_a}", flush=True
    )
    for ep in range(ep0, 0 if NUR_TEST else EPOCHEN + 1):
        if step >= steps:
            break
        if FREEZE and ep == FREEZE + 1:
            set_frozen(False)
            print(f"Epoche {ep}: vortrainierte Decoder-Teile aufgetaut", flush=True)
        for X, Y, FL, TL in tr_loader:
            X, Y, FL, TL = X.to(dev), Y.to(dev), FL.to(dev), TL.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(X)
                lp = model.ctc(h).float().log_softmax(-1)
                loss = ctc_loss(lp.permute(1, 0, 2), Y, FL, TL)
                if model.use_dec:
                    yd = ctc_to_dec.to(dev)[Y]
                    B, n = yd.shape
                    full = torch.zeros(B, n + 2, dtype=torch.long, device=dev)
                    full[:, 0] = bos
                    full[:, 1 : n + 1] = yd
                    full[torch.arange(B), TL + 1] = eos
                    valid = torch.arange(n + 2, device=dev)[None] <= (TL + 1)[:, None]
                    full = full * valid
                    pad = torch.arange(h.shape[1], device=dev)[None] >= FL[:, None]
                    logits = model.decode(full[:, :-1], model.memory(h), pad).float()
                    tgt = full[:, 1:]
                    ce = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        tgt.reshape(-1),
                        ignore_index=0,
                        label_smoothing=0.1,
                    )
                    loss = LAMBDA_CTC * loss + (1 - LAMBDA_CTC) * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(
                    f"Schritt {step}/{steps}  Verlust {loss.item():.3f}  {time.time() - t0:.0f} s",
                    flush=True,
                )
            if step >= steps:
                break
        vc, va = evaluate(model, va_loader, i2w_ctc, blank, voc, bos, eos, dev)
        rows.append((ep, step, vc, va, time.time() - t0))
        print(
            f"Epoche {ep}: Val-SER CTC {vc:.2f} %  Attention {va}  {time.time() - t0:.0f} s",
            flush=True,
        )
        crit = va if va is not None else vc
        if crit < best:
            best = crit
            torch.save(model.state_dict(), HERE / "modelle" / f"hybrid_{NAME}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "rows": rows,
                "best": best,
                "step": step,
                "ep": ep,
                "base_c": base_c,
                "base_a": base_a,
            },
            stand,
        )
        if step >= steps:
            break
    model.load_state_dict(torch.load(HERE / "modelle" / f"hybrid_{NAME}.pt"))
    tc, ta = evaluate(model, te_loader, i2w_ctc, blank, voc, bos, eos, dev)
    lines = [
        f"# Hybrid {MODUS}: Ergebnis",
        "",
        f"`skripte/hybrid.py`, MODUS={MODUS}, SEED={SEED}, LM_INIT={LM_INIT}, FREEZE={FREEZE}, ENC_LR={ENC_LR}, LR_PRE={LR_PRE}, LR_REST={lrs[2]}, Training {len(sets['train'])} Zeilen, {EPOCHEN} Epochen geplant, {step} Schritte, Batch {BATCH}, Verlust {LAMBDA_CTC} CTC + {1 - LAMBDA_CTC:.1f} Kreuzentropie. Auswahl nach Val-SER auf {len(val_idx)} Validierungszeilen. Test: alle {len(sets['test'])} Testzeilen, Greedy-Decodierung. Vor dem Training (INIT={INIT}, Daten {DATA}): Val-SER CTC {base_c:.2f} %.",
        "",
        "| Epoche | Schritte | Val-SER CTC | Val-SER Attention | Zeit (s) |",
        "|---|---|---|---|---|",
    ] + [
        f"| {e} | {s} | {c:.2f} | {'' if a is None else f'{a:.2f}'} | {t:.0f} |"
        for e, s, c, a, t in rows
    ]
    lines += [
        "",
        f"**Test-SER:** CTC {tc:.2f} %"
        + (f", Attention {ta:.2f} %" if ta is not None else "")
        + " (Bezug feste Testmenge: CRNN v3 9,56 %).",
    ]
    (HERE / "docs" / f"hybrid_{NAME}.md").write_text("\n".join(lines) + "\n")
    (HERE / "daten" / f"hybrid_{NAME}.done").write_text("ok\n")
    print("\n".join(lines))


if __name__ == "__main__":
    assert lev(list("abc"), list("abd")) == 1
    main()
