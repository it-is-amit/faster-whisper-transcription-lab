import re
import numpy as np


def ctc_forced_align(logp, ids, blank):
    """logp: (T, V) float64 log-probabilities. ids: token ids jo is order mein aane chahiye.
    Return: har token ke liye (pehla_frame, aakhri_frame), ya None agar window bahut chhoti hai."""
    T = logp.shape[0]
    n = len(ids)
    if n == 0 or T == 0:
        return None
    ids = np.asarray(ids, dtype=np.int64)
    need = n + int(np.sum(ids[1:] == ids[:-1]))      # ek jaise do akshar ke beech blank chahiye
    if T < need:
        return None
    ext = np.full(2 * n + 1, blank, dtype=np.int64)
    ext[1::2] = ids
    S = len(ext)
    NEG = -1e30
    em = logp[:, ext]                                 # (T, S)
    can_skip = np.zeros(S, dtype=bool)
    for s in range(3, S, 2):
        can_skip[s] = ext[s] != ext[s - 2]
    dp = np.full((T, S), NEG, dtype=np.float64)
    back = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = em[0, 0]
    dp[0, 1] = em[0, 1]
    ar = np.arange(S)
    for t in range(1, T):
        prev = dp[t - 1]
        c1 = np.concatenate(([NEG], prev[:-1]))
        c2 = np.concatenate(([NEG, NEG], prev[:-2]))
        c2 = np.where(can_skip, c2, NEG)
        stack = np.vstack([prev, c1, c2])
        b = stack.argmax(axis=0)
        dp[t] = stack[b, ar] + em[t]
        back[t] = b
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if dp[T - 1, s] < NEG / 2:
        return None
    states = np.empty(T, dtype=np.int64)
    for t in range(T - 1, 0, -1):
        states[t] = s
        s -= int(back[t, s])
    states[0] = s
    mask = (states % 2) == 1
    frames = np.where(mask)[0]
    toks = (states[mask] - 1) // 2
    first = np.full(n, T, dtype=np.int64)
    last = np.full(n, -1, dtype=np.int64)
    np.minimum.at(first, toks, frames)
    np.maximum.at(last, toks, frames)
    return [(int(a), int(b)) for a, b in zip(first, last)]


def line_times(spans, owner, times_win, n_lines, max_last_tok=1.5):
    """spans: har token ke (first, last) frame. owner: token kis line ka hai (-1 = space).
    times_win: window ke har frame ka second. Return: har line ke liye (start, end) ya None."""
    first_f = [None] * n_lines
    last_span = [None] * n_lines
    for k, (f, l) in enumerate(spans):
        li = owner[k]
        if li < 0:
            continue
        if first_f[li] is None:
            first_f[li] = f
        last_span[li] = (f, l)
    out = []
    for li in range(n_lines):
        if first_f[li] is None:
            out.append(None)
            continue
        f, l = last_span[li]
        l = min(l, f + int(max_last_tok / 0.02))      # lambi tan se end bahut aage na jaye
        s = float(times_win[first_f[li]])
        e = float(times_win[min(l, len(times_win) - 1)]) + 0.02
        out.append((s, max(e, s + 0.1)))
    return out


def split_lines(lyric):
    """Lyric ko lines mein todta hai. Agar sab ek hi line mein ho to danda (।) pe todta hai."""
    lines = [l.strip() for l in lyric.splitlines() if l.strip()]
    if len(lines) == 1:
        # danda ke baad todo, par "।।" ko ek hi maano
        lines = [p.strip() for p in re.split(r"(?<=[।॥])(?![।॥])\s*", lines[0]) if p.strip()]
    return [l for l in lines if re.search(r"\w", l)]      # sirf "।" wali bekaar line hata do


def srt_time(t):
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(lines, times_abs, clip_start, clip_dur, pre=0.1, tail=0.5, join_gap=1.5):
    """Subtitle ka samay clip ke shuru se naapa jata hai (clip_start = mp3 ka start).
    pre = line se thoda pehle dikhao, tail = line ke baad thoda rukne do,
    join_gap = agli line itni paas ho to is line ko agli line tak rakho."""
    items = [(txt, t) for txt, t in zip(lines, times_abs) if t is not None]
    cues = []
    for i, (txt, (s, e)) in enumerate(items):
        cs = s - clip_start - pre
        ce = e - clip_start + tail
        if i + 1 < len(items):
            ns = items[i + 1][1][0] - clip_start - pre
            if (items[i + 1][1][0] - e) <= join_gap:
                ce = ns - 0.05
            ce = min(ce, ns - 0.02)
        ce = max(ce, cs + 0.25)
        cs = max(cs, 0.0)
        ce = min(ce, clip_dur)
        if cs >= clip_dur or ce - cs < 0.15:
            continue
        cues.append((cs, ce, txt))
    out = []
    for n, (cs, ce, txt) in enumerate(cues, 1):
        out.append(f"{n}\n{srt_time(cs)} --> {srt_time(ce)}\n{txt}\n")
    return "\n".join(out)