import gc, re, subprocess
from pathlib import Path
import gradio as gr
from lyricsearch import LyricIndex, fmt

# Chalane ka tareeka:   python app.py     (phir browser mein khulta hai)
# Folder: index\<part>.npy  aur  audio\<part>.mp3  ka naam EK JAISA hona chahiye.
BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "index"
AUDIO_DIR = BASE / "audio"
CLIPS_DIR = BASE / "clips"
MAX_CLIP_SEC = 90

_state = {"part": None, "idx": None}


def list_parts():
    return sorted(p.stem for p in INDEX_DIR.glob("*.npy"))


def find_audio(part):
    for ext in (".mp3", ".MP3", ".wav", ".m4a", ".flac"):
        p = AUDIO_DIR / (part + ext)
        if p.exists():
            return p
    return None


def get_index(part):
    # ek samay mein ek hi part GPU mein rakhte hain
    if _state["part"] != part:
        _state["idx"] = None
        _state["part"] = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        _state["idx"] = LyricIndex(str(INDEX_DIR / (part + ".npy")))
        _state["part"] = part
    return _state["idx"]


def label(score):
    if score >= -2.0:
        return "PAKKA"
    if score >= -3.5:
        return "CHECK (sun ke dekh)"
    return "SHAYAD IS PART MEIN NAHI"


def cut(audio, start, end, out):
    CLIPS_DIR.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(audio), "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                    "-c:a", "libmp3lame", "-q:a", "2", str(out)],
                   check=True, capture_output=True)


def span_text(start, end):
    dur = end - start
    warn = "   (90 sec se zyada!)" if dur > MAX_CLIP_SEC else ""
    return f"{fmt(start)} - {fmt(end)}   ({dur:.1f} sec){warn}"


def do_search(part, lyric, lead_in, tail):
    if not part:
        return "Pehle part chuno.", None, None, None, ""
    if not lyric or not lyric.strip():
        return "Lyric paste karo.", None, None, None, ""
    idx = get_index(part)
    res = idx.search(lyric)
    if res is None:
        return "Lyric samajh nahi aaya (akshar nahi mile).", None, None, None, ""
    score, s, e = res
    start = float(round(max(s - lead_in, 0), 1))
    end = float(round(e + tail, 1))
    msg = (f"{label(score)}   |   score {score:.2f}\n"
           f"Gaana: {fmt(s)} - {fmt(e)}\n"
           f"Crop (gap ke saath): {span_text(start, end)}")
    return msg, start, end, None, ""


def do_preview(part, start, end):
    audio = find_audio(part) if part else None
    if audio is None:
        return None, f"audio\\{part}.mp3 nahi mili. Audio file ko audio folder mein is naam se rakho."
    if start is None or end is None or end <= start:
        return None, "Start aur end sahi daalo (end, start se bada hona chahiye)."
    out = CLIPS_DIR / "_preview.mp3"
    cut(audio, start, end, out)
    return str(out), span_text(start, end)


def do_save(part, start, end, name):
    audio = find_audio(part) if part else None
    if audio is None:
        return f"audio\\{part}.mp3 nahi mili."
    if start is None or end is None or end <= start:
        return "Start aur end sahi daalo."
    name = (name or "").strip()
    if not name:
        short = re.sub(r"[^\w\-]+", "_", part).strip("_")
        name = f"{short}_{int(start // 60)}m{int(start % 60):02d}s"
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    if not name.lower().endswith(".mp3"):
        name += ".mp3"
    out = CLIPS_DIR / name
    cut(audio, start, end, out)
    return f"Save ho gaya: clips\\{name}   ({span_text(start, end)})"


def nudge(val, delta):
    return round((val or 0) + delta, 1)


parts = list_parts()

with gr.Blocks(title="Ramayan Lyric Finder") as demo:
    gr.Markdown("## Ramayan Lyric Finder\nPart chuno, lyric paste karo, search dabao. "
                "(Abhi sirf sabse achha match dikhta hai.)")
    with gr.Row():
        part = gr.Dropdown(choices=parts, value=parts[0] if parts else None, label="Part")
        lead_in = gr.Slider(0, 5, value=2, step=0.5, label="Shuru mein extra (sec)")
        tail = gr.Slider(0, 5, value=1, step=0.5, label="End mein extra (sec)")
    lyric = gr.Textbox(lines=6, label="Lyric (book se copy karke)")
    search_btn = gr.Button("Search", variant="primary")
    status = gr.Textbox(lines=4, label="Result", interactive=False)

    with gr.Row():
        start = gr.Number(label="Start (sec)", precision=1)
        end = gr.Number(label="End (sec)", precision=1)
    with gr.Row():
        s_minus = gr.Button("Start -0.5")
        s_plus = gr.Button("Start +0.5")
        e_minus = gr.Button("End -0.5")
        e_plus = gr.Button("End +0.5")
    preview_btn = gr.Button("Sun ke dekho")
    player = gr.Audio(label="Preview", type="filepath", interactive=False)
    info = gr.Textbox(label="Samay", interactive=False)

    with gr.Row():
        name = gr.Textbox(label="File ka naam (khaali chhodo to apne aap)")
        save_btn = gr.Button("Crop karke save karo")
    saved = gr.Textbox(label="Save", interactive=False)

    search_btn.click(do_search, [part, lyric, lead_in, tail], [status, start, end, player, info])
    preview_btn.click(do_preview, [part, start, end], [player, info])
    save_btn.click(do_save, [part, start, end, name], [saved])
    s_minus.click(lambda v: nudge(v, -0.5), start, start)
    s_plus.click(lambda v: nudge(v, 0.5), start, start)
    e_minus.click(lambda v: nudge(v, -0.5), end, end)
    e_plus.click(lambda v: nudge(v, 0.5), end, end)

if __name__ == "__main__":
    demo.launch(inbrowser=True)