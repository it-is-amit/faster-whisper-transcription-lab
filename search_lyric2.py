import sys, os, re, subprocess, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoProcessor

# Use:
#   python search_lyric2.py "lyric"          -> top matches (rough + exact singing times)
#   python search_lyric2.py "lyric" crop     -> same + crop best match into clips/
AUDIO = "test.MP3"
INDEX = "test_index.npy"
MAX_CLIP_SEC = 90
LEAD_IN = 2.0      # gaane se pehle kitna music/gap rakhna hai (sec)
TAIL = 1.0         # gaane ke baad kitna extra rakhna hai (sec)
REFINE_PAD = 6.0   # barik search ke liye window ke aas-paas kitna extra dekhna hai (sec)

lyric = sys.argv[1]
do_crop = len(sys.argv) > 2 and sys.argv[2].lower() == "crop"

proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
proc.tokenizer.set_target_lang("hin")
vocab = proc.tokenizer.get_vocab()
blank = proc.tokenizer.pad_token_id

# awaaz ke hisaab se milte-julte akshar ek maano
SAME = {"श": "स", "ष": "स", "व": "ब", "ी": "ि", "ू": "ु", "ई": "इ", "ऊ": "उ",
        "ँ": "ं", "ण": "न", "ङ": "न", "ञ": "न"}
def norm(c): return SAME.get(c, c)

classes = {}
for ch, i in vocab.items():
    if len(ch) == 1:
        classes.setdefault(norm(ch), []).append(i)

# index load karo, overlap hatao, ek seedhi timeline banao
index = np.load(INDEX, allow_pickle=True)
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

# ---------- Step 1: moti search (alag-alag lambai ki windows) ----------
lens = sorted(set(int(min(n * r, MAX_CLIP_SEC) * 50) for r in np.linspace(0.2, 0.9, 8)))
lens = [L for L in lens if L > n]

results = []
tgt = torch.tensor([ids])
for a in range(0, N, 50):          # har 1 sec pe start try karo
    for L in lens:
        if a + L > N:
            continue
        seg = merged[a:a + L].unsqueeze(1)
        loss = F.ctc_loss(seg, tgt, torch.tensor([L]), torch.tensor([n]),
                          blank=blank, reduction="sum", zero_infinity=True)
        results.append((-loss.item() / n, times[a], times[a + L - 1]))

results.sort(reverse=True)

# overlap wale results ek mein jodo (ek hi jagah ke repeat hata do)
def overlaps(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    return inter / max(min(e1 - s1, e2 - s2), 1e-6) > 0.3

shown = []
for score, s, e in results:
    if not any(overlaps(s, e, x[1], x[2]) for x in shown):
        shown.append((score, s, e))
    if len(shown) == 5:
        break

# ---------- Step 2: barik alignment (exact pehla aur aakhri akshar) ----------
def ctc_align(logp, ids, blank):
    """logp: (T, V) numpy. Returns (first_frame, last_frame) of the lyric."""
    T = logp.shape[0]
    ext = [blank]
    for i in ids:
        ext += [i, blank]
    ext = np.array(ext)
    S = len(ext)
    NEG = -1e18
    em = logp[:, ext]                       # (T, S)
    allow2 = np.zeros(S, dtype=bool)
    for s in range(2, S):
        allow2[s] = (ext[s] != blank) and (ext[s] != ext[s - 2])
    dp = np.full((T, S), NEG, dtype=np.float64)
    back = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = em[0, 0]
    if S > 1:
        dp[0, 1] = em[0, 1]
    idx = np.arange(S)
    for t in range(1, T):
        prev = dp[t - 1]
        c1 = np.concatenate(([NEG], prev[:-1]))
        c2 = np.concatenate(([NEG, NEG], prev[:-2]))
        c2 = np.where(allow2, c2, NEG)
        stack = np.stack([prev, c1, c2])
        best = stack.argmax(axis=0)
        dp[t] = stack[best, idx] + em[t]
        back[t] = best
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    first = last = None
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:                      # token state
            tok = (s - 1) // 2
            if tok == len(ids) - 1 and last is None:
                last = t
            if tok == 0:
                first = t
        s -= int(back[t, s])
        if s < 0:
            break
    return first, last

def fmt(t):
    t = round(float(t), 1)
    return f"{int(t // 60)}:{t % 60:04.1f}"

final = []
for score, s, e in shown:
    a = max(int(round((s - REFINE_PAD - times[0]) * 50)), 0)
    b = min(int(round((e + REFINE_PAD - times[0]) * 50)), N)
    first, last = ctc_align(merged[a:b].numpy().astype(np.float64), ids, blank)
    if first is None or last is None:
        final.append((score, s, e, s, e))
    else:
        final.append((score, s, e, times[a + first], times[a + last]))

print()
for score, s, e, fs, fe in final:
    mark = "  (90 sec se zyada!)" if (fe - fs) > MAX_CLIP_SEC else ""
    print(f"score {score:.2f} | gaana: {fmt(fs)} - {fmt(fe)}  (rough: {fmt(s)} - {fmt(e)}){mark}")

if do_crop and final:
    score, s, e, fs, fe = final[0]
    start = max(fs - LEAD_IN, 0)
    end = fe + TAIL
    os.makedirs("clips", exist_ok=True)
    out = f"clips/clip_{int(fs)}s.mp3"
    subprocess.run(["ffmpeg", "-y", "-i", AUDIO, "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                    "-c:a", "libmp3lame", "-q:a", "2", out],
                   check=True, capture_output=True)
    print(f"\nCropped ({fmt(start)} - {fmt(end)}) -> {out}")