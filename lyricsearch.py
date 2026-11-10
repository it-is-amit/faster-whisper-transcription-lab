import re, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoProcessor

MAX_CLIP_SEC = 90
BATCH = 32      # ek saath kitni windows GPU pe (memory kam ho to ghata do)
TOP_K = 12      # moti search ke baad kitni jagah barik dekhni hain

# awaaz ke hisaab se milte-julte akshar ek maano
SAME = {"श": "स", "ष": "स", "व": "ब", "ी": "ि", "ू": "ु", "ई": "इ", "ऊ": "उ",
        "ँ": "ं", "ण": "न", "ङ": "न", "ञ": "न"}
def norm(c): return SAME.get(c, c)


def fmt(t):
    if t is None:
        return "  ?  "
    t = round(float(t), 1)
    return f"{int(t // 60)}:{t % 60:04.1f}"


class LyricIndex:
    """Ek part ka index load karta hai, phir kitne bhi lyric search kar sakte ho."""

    def __init__(self, index_path):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        proc.tokenizer.set_target_lang("hin")
        vocab = proc.tokenizer.get_vocab()
        self.blank = proc.tokenizer.pad_token_id

        classes = {}
        for ch, i in vocab.items():
            if len(ch) == 1:
                classes.setdefault(norm(ch), []).append(i)

        # index load karo, overlap hatao, ek seedhi timeline banao
        index = np.load(index_path, allow_pickle=True)
        frames, times = [], []
        for k, (s, lp) in enumerate(index):
            T = lp.shape[0]
            lo = 0 if k == 0 else 50
            hi = T if k == len(index) - 1 else T - 50
            frames.append(lp[lo:hi])
            times.append(s + np.arange(lo, hi) * 0.02)
        lp = torch.tensor(np.concatenate(frames))
        self.times = np.concatenate(times)
        self.N, V = lp.shape

        # milte-julte tokens ki probability jod do
        merged = torch.full((self.N, V), -1e9)
        self.rep = {}
        for c, ids_ in classes.items():
            r = vocab.get(c, ids_[0])
            self.rep[c] = r
            merged[:, r] = torch.logsumexp(lp[:, ids_], dim=1)
        merged[:, self.blank] = lp[:, self.blank]
        self.merged = merged.to(self.dev)

    def tokens(self, lyric):
        clean = re.sub(r"[।॥,.!?;:\-\d]", " ", lyric)
        text = "|".join(clean.split())
        return [self.rep[norm(c)] for c in text if norm(c) in self.rep]

    def _scores(self, ids, starts, L):
        """Har start pe L frame ki window ka score (jitna zero ke paas, utna achha match)."""
        n = len(ids)
        out = np.empty(len(starts), dtype=np.float32)
        tgt = torch.tensor([ids], device=self.dev)
        ar = torch.arange(L, device=self.dev)
        for i in range(0, len(starts), BATCH):
            st = torch.as_tensor(np.asarray(starts[i:i + BATCH]), device=self.dev)
            B = st.shape[0]
            seg = self.merged[st[:, None] + ar[None, :]]       # (B, L, V)
            seg = seg.permute(1, 0, 2).contiguous()            # (L, B, V)
            loss = F.ctc_loss(seg, tgt.expand(B, -1),
                              torch.full((B,), L, dtype=torch.long),
                              torch.full((B,), n, dtype=torch.long),
                              blank=self.blank, reduction="none", zero_infinity=True)
            out[i:i + B] = (-loss / n).cpu().numpy()
        return out

    def search(self, lyric, fast=True):
        """Best match: (score, start_sec, end_sec). fast=False = poori (dheemi) search."""
        ids = self.tokens(lyric)
        n = len(ids)
        if n == 0:
            return None
        N = self.N
        # lambaiyan 2 se 90 sec, har 2 sec ka step (gaane ki speed pe depend nahi)
        lens = [int(sec * 50) for sec in np.arange(2, MAX_CLIP_SEC + 0.1, 2.0)]
        lens = [L for L in lens if L > n * 1.5 and L <= N]
        if not lens:
            return None

        best = None

        def consider(score, a, L):
            nonlocal best
            if best is None or score > best[0]:
                best = (float(score), self.times[a], self.times[a + L - 1])

        if not fast:
            # poori search: har 1 sec pe start, har lambai
            for L in lens:
                starts = np.arange(0, N - L + 1, 50)
                sc = self._scores(ids, starts, L)
                j = int(sc.argmax())
                consider(sc[j], int(starts[j]), L)
            return best

        # --- Step 1: moti search (har 3 sec pe start, har 4 sec ki lambai) ---
        clens = [L for L in lens if L % 200 == 0] or lens
        cand = []
        for L in clens:
            starts = np.arange(0, N - L + 1, 150)
            sc = self._scores(ids, starts, L)
            cand += [(float(v), int(a), L) for a, v in zip(starts, sc)]
        cand.sort(reverse=True)
        picked = []
        for v, a, L in cand:
            if all(abs(a - p[1]) >= 300 for p in picked):     # 6 sec se door ki alag jagah
                picked.append((v, a, L))
            if len(picked) == TOP_K:
                break

        # --- Step 2: sirf un jagah ke aas-paas poori barik grid (wahi grid jo poori search mein hai) ---
        for v0, a0, L0 in picked:
            for L in lens:
                if abs(L - L0) > 300:
                    continue
                starts = np.arange(a0 - 300, a0 + 301, 50)
                starts = starts[(starts >= 0) & (starts <= N - L)]
                if len(starts) == 0:
                    continue
                sc = self._scores(ids, starts, L)
                j = int(sc.argmax())
                consider(sc[j], int(starts[j]), L)
        return best