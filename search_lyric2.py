import sys, re, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoProcessor

lyric = sys.argv[1]
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
proc.tokenizer.set_target_lang("hin")
vocab = proc.tokenizer.get_vocab()
blank = proc.tokenizer.pad_token_id

# awaaz ke hisaab se milte-julte akshar ek maano
SAME = {"श":"स","ष":"स","व":"ब","ी":"ि","ू":"ु","ई":"इ","ऊ":"उ",
        "ँ":"ं","ण":"न","ङ":"न","ञ":"न"}
def norm(c): return SAME.get(c, c)

# model ke tokens ko classes mein jodo
classes = {}
for ch, i in vocab.items():
    if len(ch) == 1:
        classes.setdefault(norm(ch), []).append(i)

# index load karo, overlap hatao, ek seedhi timeline banao
index = np.load("test_index.npy", allow_pickle=True)
frames, times = [], []
for k, (s, lp) in enumerate(index):
    T = lp.shape[0]
    lo = 0 if k == 0 else 50
    hi = T if k == len(index) - 1 else T - 50
    frames.append(lp[lo:hi])
    times.append(s + np.arange(lo, hi) * 0.02)
lp = torch.tensor(np.concatenate(frames))
times = np.concatenate(times)
N, V = lp.shape

# milte-julte tokens ki probability jod do
merged = torch.full((N, V), -1e9)
rep = {}
for c, ids_ in classes.items():
    r = vocab.get(c, ids_[0])
    rep[c] = r
    merged[:, r] = torch.logsumexp(lp[:, ids_], dim=1)
merged[:, blank] = lp[:, blank]

# lyric saaf karo
clean = re.sub(r"[।॥,.!?;:\-\d]", " ", lyric)
text = "|".join(clean.split())
ids = [rep[norm(c)] for c in text if norm(c) in rep]
n = len(ids)
print(f"lyric tokens: {n}")

# alag-alag lambai ki windows (0.2 se 0.9 sec per akshar, max 90 sec)
lens = sorted(set(int(min(n * r, 90) * 50) for r in np.linspace(0.2, 0.9, 8)))
lens = [L for L in lens if L > n]

results = []
tgt = torch.tensor([ids])
for a in range(0, N, 50):          # har 1 sec pe start try karo
    for L in lens:
        if a + L > N: continue
        seg = merged[a:a+L].unsqueeze(1)
        loss = F.ctc_loss(seg, tgt, torch.tensor([L]), torch.tensor([n]),
                          blank=blank, reduction="sum", zero_infinity=True)
        results.append((-loss.item() / n, times[a], times[a+L-1]))

results.sort(reverse=True)
shown = []
for score, s, e in results:
    if all(abs(s - x) > 10 for x in shown):
        shown.append(s)
        mark = "  (90 sec se zyada!)" if e - s > 90 else ""
        print(f"score {score:.2f} | {int(s//60)}:{s%60:04.1f} - {int(e//60)}:{e%60:04.1f}{mark}")
    if len(shown) == 5: break