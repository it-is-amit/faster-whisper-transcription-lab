from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

model_id = "facebook/mms-1b-all"
processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

processor.tokenizer.set_target_lang("hin")
model.load_adapter("hin")

model = model.half().to("cuda")
print("Model ready on GPU")
print("GPU memory used (GB):", torch.cuda.memory_allocated() / 1e9)