import time, argparse
from lyricsearch import LyricIndex, fmt

# Use:
#   python batch_test.py
#   python batch_test.py --compare        (tez search ko poori dheemi search se bhi milata hai)
#   python batch_test.py --tests tests.txt --index "index\Part4.npy"
START_EARLY_OK = 5.0   # tool ka start itna pehle ho to theek (extra music/gap)
START_LATE_OK = 1.5    # tool ka start itna late ho to theek (isse zyada = pehla shabd kat sakta hai)
END_LATE_OK = 4.0      # tool ka end itna baad ho to theek (aakhri sur ke liye)
END_EARLY_OK = 1.5     # tool ka end itna pehle ho to theek (isse zyada = aakhri shabd kat sakta hai)
ABSENT_SCORE = -3.0    # "lyric nahi hai" wale test mein isse kamzor score = sahi (nahi mila)

ap = argparse.ArgumentParser()
ap.add_argument("--tests", default="tests.txt")
ap.add_argument("--index", default=r"index\Ravindra Ramayan - Bal Kand Part 4.npy")
ap.add_argument("--compare", action="store_true")
args = ap.parse_args()

# ---------- tests.txt padho ----------
def parse_time(t):
    if t in ("?", "-"):
        return None
    sec = 0.0
    for p in t.split(":"):
        sec = sec * 60 + float(p)
    return sec

tests = []
for line in open(args.tests, encoding="utf-8-sig"):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("@"):
        parts = line[1:].split()
        tests.append({"start": parse_time(parts[0]), "end": parse_time(parts[1]),
                      "absent": parts[0] == "-", "lyric": ""})
    elif tests:
        tests[-1]["lyric"] += " " + line
print(f"{len(tests)} tests mile")

t_start = time.time()
idx = LyricIndex(args.index)
print(f"index load: {time.time() - t_start:.0f} sec")

# ---------- saare tests chalao ----------
ok_count = 0
same_count = 0
search_time = 0.0
for i, tc in enumerate(tests, 1):
    t1 = time.time()
    res = idx.search(tc["lyric"])
    search_time += time.time() - t1
    preview = tc["lyric"].strip()[:16]
    if res is None:
        print(f"{i:>2}. {preview:<18} lyric samajh nahi aaya")
        continue
    score, s, e = res

    cmp_txt = ""
    if args.compare:
        full = idx.search(tc["lyric"], fast=False)
        same = abs(full[1] - s) < 0.01 and abs(full[2] - e) < 0.01
        same_count += same
        cmp_txt = " | poori search se " + ("SAME" if same else f"ALAG ({fmt(full[1])}-{fmt(full[2])}, score {full[0]:.2f})")

    if tc["absent"]:
        good = score < ABSENT_SCORE
        verdict = "OK (nahi mila)" if good else "GALAT: strong match mila"
        print(f"{i:>2}. {preview:<18} lyric hai hi nahi | found {fmt(s)}-{fmt(e)} | score {score:.2f} | {verdict}{cmp_txt}")
        ok_count += good
        continue

    problems = []
    ds = de = None
    if tc["start"] is not None:
        ds = s - tc["start"]
        if ds < -START_EARLY_OK: problems.append("start bahut pehle")
        if ds > START_LATE_OK: problems.append("start late (shabd kat sakta hai)")
    if tc["end"] is not None:
        de = e - tc["end"]
        if de > END_LATE_OK: problems.append("end bahut baad")
        if de < -END_EARLY_OK: problems.append("end pehle (shabd kat sakta hai)")
    good = not problems
    ok_count += good
    d1 = f"{ds:+.1f}s" if ds is not None else "  ?  "
    d2 = f"{de:+.1f}s" if de is not None else "  ?  "
    verdict = "OK" if good else "CHECK: " + ", ".join(problems)
    print(f"{i:>2}. {preview:<18} tera {fmt(tc['start'])}-{fmt(tc['end'])} | tool {fmt(s)}-{fmt(e)} "
          f"| start {d1} end {d2} | score {score:.2f} | {verdict}{cmp_txt}")

print(f"\n{ok_count} / {len(tests)} OK")
if args.compare:
    print(f"tez search = poori search: {same_count} / {len(tests)}")
print(f"search ka time (tez wali, sab milake): {search_time:.0f} sec   |   total: {time.time() - t_start:.0f} sec")