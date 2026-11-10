import sys, os, time, subprocess, numpy as np, torch, soundfile as sf
from transformers import Wav2Vec2ForCTC, AutoProcessor

# Use:
#   python index_audio2.py "S:\path\to\BalKand_Part4.mp3"
# Result: index\BalKand_Part4.npy  (ek baar banta hai, phir search ke kaam aata hai)

src = sys.argv[1]
name = os.path.splitext(os.path.basename(src))[0]
os.makedirs("index", exist_ok=True)
out_path = os.path.join("index", name + ".npy")
wav_path = os.path.join("index", name + "_16k.wav")

t0 = time.time()
print("audio ko 16kHz mono mein badal raha hoon...")
subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", wav_path],
               check=True, capture_output=True)
audio, sr = sf.read(wav_path, dtype="float32")
print(f"audio length: {len(audio) / sr / 60:.1f} min")

model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
processor.tokenizer.set_target_lang("hin")
model.load_adapter("hin")
model = model.half().to("cuda").eval()

chunk, overlap = 30 * sr, 2 * sr   # 30 sec ke tukde, 2 sec overlap
step = chunk - overlap
out = []
total = len(audio)
for i, s in enumerate(range(0, total, step)):
    piece = audio[s:s + chunk]
    if len(piece) < sr:            # 1 sec se chhota tukda skip (pichla tukda cover kar chuka hai)
        continue
    inp = processor(piece, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        lp = model(inp.input_values.half().to("cuda")).logits.float().log_softmax(-1)[0].cpu().numpy()
    out.append((s / sr, lp))
    if i % 10 == 0:
        print(f"{s / sr / 60:.1f} / {total / sr / 60:.1f} min  ({time.time() - t0:.0f}s laga)")

np.save(out_path, np.array(out, dtype=object), allow_pickle=True)
os.remove(wav_path)
print(f"Saved {out_path}  |  total time: {(time.time() - t0) / 60:.1f} min")
