import sys, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoProcessor

lyric = sys.argv[1]
processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
processor.tokenizer.set_target_lang("hin")
vocab = processor.tokenizer.get_vocab()

# lyric ko model ke tokens mein badlo ("|" = space)
text = " ".join(lyric.replace("।", " ").replace(",", " ").split()).replace(" ", "|")
ids = [vocab[c] for c in text if c in vocab]
print("tokens:", len(ids), "of", len(text), "chars")

index = np.load("test_index.npy", allow_pickle=True)
FPS = 50  # 1 frame = 0.02 sec
results = []
for start_s, lp in index:
    lp_t = torch.tensor(lp)
    T = lp_t.shape[0]
    win = int(max(len(ids) * 2.5, 100))  # lyric ke hisaab se window
    hop = 25
    for a in range(0, max(T - win, 1), hop):
        seg = lp_t[a:a+win]
        loss = F.ctc_loss(seg.unsqueeze(1), torch.tensor([ids]),
                          torch.tensor([seg.shape[0]]), torch.tensor([len(ids)]),
                          blank=processor.tokenizer.pad_token_id,
                          reduction="sum", zero_infinity=True)
        results.append((-loss.item() / len(ids), start_s + a / FPS, start_s + (a + win) / FPS))

results.sort(reverse=True)
shown = []
for score, s, e in results:
    if all(abs(s - x) > 10 for x in shown):
        shown.append(s)
        print(f"score {score:.2f} | {int(s//60)}:{s%60:04.1f} - {int(e//60)}:{e%60:04.1f}")
    if len(shown) == 5:
        break