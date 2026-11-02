from __future__ import annotations
import torch
import torch.nn as nn
from itertools import groupby

D, LAYERS, HEADS, FF, MAXLEN = 256, 6, 4, 1024, 224
MAXOUT = 230
MODUS = 'scratch'

class glob_vars:
    FIXED_HEIGHT = 128
    LAST_LAYER_DEPTH_OMR = 256



class CRNN_OMR(nn.Module):
    def __init__(self, output_projection):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(128, glob_vars.LAST_LAYER_DEPTH_OMR, kernel_size=3, padding="same"),
            nn.BatchNorm2d(glob_vars.LAST_LAYER_DEPTH_OMR),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        self.rnn_features = (
            glob_vars.FIXED_HEIGHT // (2**4)
        ) * glob_vars.LAST_LAYER_DEPTH_OMR

        self.rnn = nn.Sequential(
            nn.LSTM(
                input_size=self.rnn_features,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
            )
        )

        self.vocab_projection = nn.Linear(512, output_projection)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], self.rnn_features)
        x, _ = self.rnn(x)
        x = self.vocab_projection(x)
        x = self.softmax(x)
        x = x.permute(1, 0, 2).contiguous()
        return x


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
    for _ in range(MAXOUT):
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
    n = max(map(len, seqs)) + 2
    x = torch.zeros(len(seqs), n, dtype=torch.long, device=dev)
    for k, s in enumerate(seqs):
        x[k, : len(s) + 2] = torch.tensor([bos] + s + [eos])
    lp = lm(x[:, :-1]).float().log_softmax(-1)
    tgt = x[:, 1:]
    return (lp.gather(-1, tgt[..., None])[..., 0] * (tgt != 0)).sum(1).tolist()
