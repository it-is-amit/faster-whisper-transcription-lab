import json
import re
from rapidfuzz import fuzz

def normalize(text):
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

with open("lyrics.txt", encoding="utf-8") as f:
    lyrics = [normalize(line) for line in f if line.strip()]

with open("transcript.json", encoding="utf-8") as f:
    segments = json.load(f)

for index, lyric in enumerate(lyrics, 1):
    results = []

    for segment in segments:
        transcript = normalize(segment["text"])
        score = fuzz.ratio(lyric, transcript)

        results.append({
            "score": score,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\nChaupai {index}")

    for result in results[:3]:
        print(
            f"Score: {result['score']:.2f}% | "
            f"{result['start']:.2f}s - {result['end']:.2f}s"
        )