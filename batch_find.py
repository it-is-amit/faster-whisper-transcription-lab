import os, sys, argparse, subprocess, time
from lyricsearch import LyricIndex, fmt

# Use:
#   python batch_find.py chaupai_list.txt --index "index\Part4.npy"
#   python batch_find.py chaupai_list.txt --index "index\Part4.npy" --audio "Part4.mp3" --crop
# chaupai_list.txt: har lyric ek block, blocks ke beech mein khaali line.
LEAD_IN = 2.0
TAIL = 1.0

ap = argparse.ArgumentParser()
ap.add_argument("lyrics_file")
ap.add_argument("--index", default=r"index\Ravindra Ramayan - Bal Kand Part 4.npy")
ap.add_argument("--audio", default=None, help="original mp3 (--crop ke liye)")
ap.add_argument("--crop", action="store_true", help="har match ko clips/ mein crop karo, sun ke check karne ke liye")
args = ap.parse_args()

# lyrics padho: khaali line se alag block = ek lyric
blocks, cur = [], []
for line in open(args.lyrics_file, encoding="utf-8-sig"):
    line = line.strip()
    if line.startswith("#"):
        continue
    if line:
        cur.append(line)
    elif cur:
        blocks.append(" ".join(cur)); cur = []
if cur:
    blocks.append(" ".join(cur))
print(f"{len(blocks)} lyrics mile")

t0 = time.time()
idx = LyricIndex(args.index)
print(f"index load: {time.time() - t0:.0f} sec\n")

if args.crop:
    if not args.audio:
        sys.exit("--crop ke saath --audio bhi do")
    os.makedirs("clips", exist_ok=True)

prev_start = None
counts = {"pakka": 0, "check": 0, "nahi": 0}
order_bad = 0
for i, lyric in enumerate(blocks, 1):
    res = idx.search(lyric)
    preview = lyric[:18]
    if res is None:
        print(f"{i:>2}. {preview:<20} lyric samajh nahi aaya")
        continue
    score, s, e = res
    if score >= -2.0:
        tag = "pakka"
    elif score >= -3.5:
        tag = "check"
    else:
        tag = "nahi"
    counts[tag] += 1
    label = {"pakka": "PAKKA", "check": "CHECK (sun ke dekh)", "nahi": "SHAYAD IS PART MEIN NAHI"}[tag]

    order = ""
    if tag != "nahi":                       # kamzor match order mein ginti nahi
        if prev_start is not None and s < prev_start - 1.0:
            order = "  | ORDER ULTA (pichli se pehle aaya)"
            order_bad += 1
        prev_start = s
    print(f"{i:>2}. {preview:<20} {fmt(s)} - {fmt(e)} | score {score:.2f} | {label}{order}")

    if args.crop and tag != "nahi":
        start = max(s - LEAD_IN, 0)
        end = e + TAIL
        out = f"clips/{i:02d}_{int(s)}s.mp3"
        subprocess.run(["ffmpeg", "-y", "-i", args.audio, "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                        "-c:a", "libmp3lame", "-q:a", "2", out],
                       check=True, capture_output=True)

print(f"\npakka: {counts['pakka']} | check: {counts['check']} | shayad nahi: {counts['nahi']} | order ulta: {order_bad}")
print(f"total time: {time.time() - t0:.0f} sec")