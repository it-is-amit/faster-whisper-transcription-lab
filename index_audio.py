import torch, numpy as np, subprocess
from transformers import Wav2Vec2ForCTC, AutoProcessor

# audio ko 16kHz mono mein badlo (ffmpeg se)
subprocess.run(["ffmpeg","-y","-i","test.MP3","-ar","16000","-ac","1","test_16k.wav"],
               check=True, capture_output=True)
import soundfile as sf
audio, sr = sf.read("test_16k.wav", dtype="float32")

model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)
processor.tokenizer.set_target_lang("hin")
model.load_adapter("hin")
model = model.half().to("cuda").eval()

chunk, overlap = 30*sr, 2*sr   # 30 sec ke tukde, 2 sec overlap
step = chunk - overlap
out = []
for s in range(0, len(audio), step):
    piece = audio[s:s+chunk]
    inp = processor(piece, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        lp = model(inp.input_values.half().to("cuda")).logits.float().log_softmax(-1)[0].cpu().numpy()
    out.append((s/sr, lp))
    print(f"done {s/sr:.0f}s")

np.save("test_index.npy", np.array(out, dtype=object), allow_pickle=True)
print("Saved test_index.npy")