import numpy as np
from transformers import AutoProcessor

p = AutoProcessor.from_pretrained("facebook/mms-1b-all")
p.tokenizer.set_target_lang("hin")
inv = {v: k for k, v in p.tokenizer.get_vocab().items()}
blank = p.tokenizer.pad_token_id

index = np.load("test_index.npy", allow_pickle=True)
a, b = 535, 560   # 8:55 se 9:20
text, prev = "", None
for start_s, lp in index:
    for i, t in enumerate(lp.argmax(-1)):
        sec = start_s + i * 0.02
        if a <= sec <= b and t != prev and t != blank:
            text += inv[int(t)]
        prev = t
print(text.replace("|", " "))