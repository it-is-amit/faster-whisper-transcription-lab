import json
from faster_whisper import WhisperModel

print("Loading Whisper model on GPU...")

model = WhisperModel(
    "medium",
    device="cuda",
    compute_type="float16"
)

print("Transcribing audio...")

segments, info = model.transcribe(
    "test.mp3",
    language="hi",
    beam_size=5
)

transcript = []

for segment in segments:
    transcript.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip()
    })

with open("transcript.json", "w", encoding="utf-8") as file:
    json.dump(transcript, file, ensure_ascii=False, indent=4)

print("Transcript saved successfully!")