"""CopyLM: encoder + action decoder with span-copy pointers.

Each decoder step emits one action:
  GEN  -> one character from the vocabulary
  COPY -> a (start, end) pair of positions in the input; the whole span is pasted
  EOS  -> stop
A pasted span is fed back to the decoder as one pooled vector, so a 1,000-char paste
costs one decoding step instead of 1,000.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import BOS, COPY, EOS, GEN, VOCAB_SIZE, encode, itos


class CopyLM(nn.Module):
    def __init__(self, E: int = 64, H: int = 192, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        D = 2 * H
        self.emb = nn.Embedding(vocab_size, E, padding_idx=0)
        self.enc = nn.GRU(E, H, batch_first=True, bidirectional=True)
        self.span_proj = nn.Linear(D, E)  # pooled pasted span -> decoder input
        self.type_emb = nn.Embedding(4, E)  # GEN / COPY / EOS / BOS
        self.init = nn.Linear(D, H)
        self.dec = nn.GRU(E, H, batch_first=True)
        self.q_ctx = nn.Linear(H, D)
        self.type_head = nn.Linear(H + D, 3)
        self.char_head = nn.Linear(H + D, vocab_size)
        self.q_start = nn.Linear(H + D, D)
        self.q_end = nn.Linear(H + D, D)

    # ---- building blocks -------------------------------------------------
    def encode(self, x):
        mask = x != 0
        enc, _ = self.enc(self.emb(x))
        m = mask.unsqueeze(-1).float()
        h0 = torch.tanh(self.init((enc * m).sum(1) / m.sum(1)))
        return enc, mask, h0

    @staticmethod
    def span_mean(enc, st, en):
        """Mean of encoder states over inclusive spans. st, en: (B, T)."""
        cs = F.pad(enc.cumsum(1), (0, 0, 1, 0))
        D = enc.size(-1)

        def g(i):
            return cs.gather(1, i.unsqueeze(-1).expand(-1, -1, D))

        return (g(en + 1) - g(st)) / (en - st + 1).unsqueeze(-1).float()

    def act_emb(self, ptype, pchar, pspan):
        """Embed the previous action (type + char or pooled span)."""
        e = self.type_emb(ptype)
        e = e + self.emb(pchar) * (ptype == GEN).unsqueeze(-1)
        e = e + self.span_proj(pspan) * (ptype == COPY).unsqueeze(-1)
        return e

    def heads(self, out, enc, mask):
        sc = torch.einsum("btd,bld->btl", self.q_ctx(out), enc).masked_fill(~mask.unsqueeze(1), -1e9)
        ctx = torch.einsum("btl,bld->btd", sc.softmax(-1), enc)
        f = torch.cat([out, ctx], -1)

        def ptr(q):
            return torch.einsum("btd,bld->btl", q(f), enc).masked_fill(~mask.unsqueeze(1), -1e9)

        return self.type_head(f), self.char_head(f), ptr(self.q_start), ptr(self.q_end)

    # ---- training --------------------------------------------------------
    def forward(self, x, ty, ch, st, en):
        enc, mask, h0 = self.encode(x)
        B, _ = ty.shape
        zeros = torch.zeros(B, 1, dtype=torch.long)
        # teacher forcing: feed the previous gold action
        pty = torch.cat([torch.full((B, 1), BOS), ty[:, :-1].clamp(min=0)], 1)
        pch = torch.cat([zeros, ch[:, :-1].clamp(min=0)], 1)
        pst = torch.cat([zeros, st[:, :-1].clamp(min=0)], 1)
        pen = torch.cat([zeros, en[:, :-1].clamp(min=0)], 1)
        pspan = self.span_mean(enc, pst, torch.maximum(pen, pst))
        out, _ = self.dec(self.act_emb(pty, pch, pspan), h0.unsqueeze(0))
        lt, lc, ls, le = self.heads(out, enc, mask)

        def ce(logits, target):
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100)

        return ce(lt, ty) + ce(lc, ch) + ce(ls, st) + ce(le, en)

    # ---- inference -------------------------------------------------------
    @torch.no_grad()
    def generate(self, inp: str, max_steps: int = 12):
        """Greedy decoding. Returns (output_text, trace) where trace lists the actions taken."""
        x = torch.tensor([encode(inp)])
        enc, mask, h = self.encode(x)
        h = h.unsqueeze(0)
        D = enc.size(-1)
        ptype = torch.tensor([[BOS]])
        pchar = torch.zeros(1, 1, dtype=torch.long)
        pspan = torch.zeros(1, 1, D)
        text, trace = "", []
        for _ in range(max_steps):
            o, h = self.dec(self.act_emb(ptype, pchar, pspan), h)
            lt, lc, ls, le = self.heads(o, enc, mask)
            t = lt.argmax(-1).item()
            if t == EOS:
                break
            if t == GEN:
                cid = lc.argmax(-1)
                c = itos.get(cid.item(), "?")
                text += c
                trace.append(f"GEN({c!r})")
                ptype, pchar, pspan = torch.tensor([[GEN]]), cid, torch.zeros(1, 1, D)
            else:
                a = ls.argmax(-1).item()
                le = le.clone()
                le[0, 0, :a] = -1e9  # constrain end >= start
                b = le.argmax(-1).item()
                text += inp[a: b + 1]
                trace.append(f"COPY[{a}:{b}]")
                sp = self.span_mean(enc, torch.tensor([[a]]), torch.tensor([[b]]))
                ptype, pchar, pspan = torch.tensor([[COPY]]), torch.zeros(1, 1, dtype=torch.long), sp
        return text, trace
