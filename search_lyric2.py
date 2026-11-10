import os, argparse, subprocess, time
from lyricsearch import LyricIndex, fmt

# Use:
#   python search_lyric2.py "lyric" --index "index\Part4.npy"
#   python search_lyric2.py "lyric" --index "index\Part4.npy" --audio "Part4.mp3" --crop
LEAD_IN = 2.0      # gaane se pehle kitna music/gap rakhna hai (sec)
TAIL = 1.0         # gaane ke baad kitna extra rakhna hai (sec)

ap = argparse.ArgumentParser()
ap.add_argument("lyric")
ap.add_argument("--index", default="test_index.npy", help="index .npy file")
ap.add_argument("--audio", default="test.MP3", help="original mp3 (crop ke liye)")
ap.add_argument("--crop", action="store_true", help="best match ko clips/ mein crop karo")
args = ap.parse_args()

t0 = time.time()
idx = LyricIndex(args.index)
t1 = time.time()
res = idx.search(args.lyric)
t2 = time.time()

if res is None:
    print("lyric samajh nahi aaya (akshar nahi mile)")
    raise SystemExit

score, s, e = res
note = "  (kamzor match, sun ke check kar)" if score < -3 else ""
print(f"score {score:.2f} | {fmt(s)} - {fmt(e)}{note}")
print(f"time laga: load {t1 - t0:.0f} sec + search {t2 - t1:.0f} sec")

if args.crop:
    start = max(s - LEAD_IN, 0)
    end = e + TAIL
    os.makedirs("clips", exist_ok=True)
    out = f"clips/clip_{int(s)}s.mp3"
    subprocess.run(["ffmpeg", "-y", "-i", args.audio, "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                    "-c:a", "libmp3lame", "-q:a", "2", out],
                   check=True, capture_output=True)
    print(f"Cropped ({fmt(start)} - {fmt(end)}) -> {out}")