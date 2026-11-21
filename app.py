import gc, re, subprocess
from pathlib import Path
import gradio as gr
from lyricsearch import LyricIndex, fmt
from srtalign import split_lines, build_srt

# Chalane ka tareeka:   start.bat par double-click   (ya:  python app.py)
# Folder: index\<part>.npy  aur  audio\<part>.mp3  ka naam EK JAISA hona chahiye.
BASE = Path(__file__).resolve().parent
INDEX_DIR = BASE / "index"
AUDIO_DIR = BASE / "audio"
CLIPS_DIR = BASE / "clips"
MAX_CLIP_SEC = 90

_state = {"part": None, "idx": None}


# ---------------------------------------------------------------- helpers
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


def verdict(score):
    if score >= -2.0:
        return "✅ **Pakka mila**"
    if score >= -3.5:
        return "⚠️ **Mila, par sun ke check karo**"
    return "❌ **Shayad ye lyric is part mein hai hi nahi**"


def cut(audio, start, end, out):
    CLIPS_DIR.mkdir(exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(audio), "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                        "-c:a", "libmp3lame", "-q:a", "2", str(out)],
                       check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg nahi mila (PATH mein nahi hai).")
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg error: " + e.stderr.decode(errors="ignore")[-300:])


def span_text(start, end):
    dur = end - start
    warn = "  ⚠️ 90 sec se zyada!" if dur > MAX_CLIP_SEC else ""
    return f"Clip: **{fmt(start)} - {fmt(end)}**  ({dur:.1f} sec){warn}"


def clip_base(part, start, name):
    """mp3 aur srt dono ka ek hi naam (bina extension)."""
    name = (name or "").strip()
    if not name:
        short = re.sub(r"[^\w\-]+", "_", part).strip("_")
        name = f"{short}_{int(start // 60)}m{int(start % 60):02d}s"
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    for ext in (".mp3", ".srt"):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
    return name


def valid_range(start, end):
    return start is not None and end is not None and end > start


# ---------------------------------------------------------------- actions
def do_search(part, lyric, lead_in, tail):
    # outputs: result, start, end, player, info, win(state), srt_box, srt_info, srt_meta(state)
    empty = (None, None, None, "", None, "", "", None)
    if not part:
        return ("❗ Pehle part chuno.",) + empty
    if not lyric or not lyric.strip():
        return ("❗ Lyric paste karo.",) + empty
    res = get_index(part).search(lyric)
    if res is None:
        return ("❗ Lyric samajh nahi aaya (akshar nahi mile).",) + empty
    score, s, e = res
    start = float(round(max(s - lead_in, 0), 1))   # numpy nahi, seedha Python float
    end = float(round(e + tail, 1))
    msg = f"{verdict(score)}  \nGaana: {fmt(s)} - {fmt(e)}   |   score {score:.2f}"
    return msg, start, end, None, "", {"s": float(s), "e": float(e)}, "", "", None


def do_preview(part, start, end):
    # outputs: player, info
    if start is None and end is None:
        return None, ""                              # abhi search hua hi nahi
    audio = find_audio(part) if part else None
    if audio is None:
        return None, f"❗ `audio\\{part}.mp3` nahi mili. Audio file ko audio folder mein isi naam se rakho."
    if not valid_range(start, end):
        return None, "❗ Start aur End sahi daalo (End, Start se bada hona chahiye)."
    out = CLIPS_DIR / "_preview.mp3"
    try:
        cut(audio, start, end, out)
    except RuntimeError as e:
        return None, f"❗ {e}"
    return str(out), span_text(start, end)


def do_srt(part, lyric, start, end, win):
    # outputs: srt_box, srt_info, srt_meta(state)
    if start is None and end is None:
        return "", "", None
    if not part or not lyric or not lyric.strip():
        return "", "❗ Part aur lyric chahiye.", None
    if not valid_range(start, end):
        return "", "❗ Start aur End sahi daalo.", None
    lines = split_lines(lyric)
    if not lines:
        return "", "❗ Lyric mein koi line nahi mili.", None
    ws, we = (win["s"], win["e"]) if win else (start, end)
    times = get_index(part).align_lines(lines, ws, we)
    if times is None:
        return "", "❗ Lines ka samay nahi nikal paya (window bahut chhoti hai ya lyric nahi mila).", None
    srt = build_srt(lines, times, start, end - start)
    info = (f"{len(lines)} lines mein se {srt.count('-->')} subtitle bane. "
            f"Samay clip ke shuru se naapa hua hai. Yahan edit kar sakte ho.")
    return srt, info, {"clip_start": float(start)}


def do_save_all(part, lyric, start, end, name, win, srt_text, srt_meta):
    # outputs: saved, srt_box, srt_meta(state)
    audio = find_audio(part) if part else None
    if audio is None:
        return f"❗ `audio\\{part}.mp3` nahi mili.", srt_text, srt_meta
    if not valid_range(start, end):
        return "❗ Pehle lyric dhundho (Start aur End sahi hone chahiye).", srt_text, srt_meta
    base = clip_base(part, start, name)
    try:
        cut(audio, start, end, CLIPS_DIR / (base + ".mp3"))
    except RuntimeError as e:
        return f"❗ {e}", srt_text, srt_meta
    msgs = [f"✅ Clip save hua: `clips\\{base}.mp3`  ({end - start:.1f} sec)"]

    fresh = bool(srt_text and srt_text.strip() and srt_meta and abs(srt_meta["clip_start"] - start) <= 0.05)
    if not fresh:                                    # SRT nahi bani thi, ya Start badal gaya
        srt_text, info, srt_meta = do_srt(part, lyric, start, end, win)
        if not srt_text:
            msgs.append(f"⚠️ SRT nahi ban paya. {info}")
            return "\n\n".join(msgs), srt_text, srt_meta
        msgs.append("(SRT clip ke naye Start ke hisaab se dobara banayi gayi)")
    (CLIPS_DIR / (base + ".srt")).write_text(srt_text.strip() + "\n", encoding="utf-8")
    msgs.append(f"✅ SRT save hui: `clips\\{base}.srt`")
    return "\n\n".join(msgs), srt_text, srt_meta


def nudge(val, delta):
    return round((val or 0) + delta, 1)


# ---------------------------------------------------------------- screen
import inspect

try:
    THEME = gr.themes.Soft(primary_hue="orange", secondary_hue="amber", neutral_hue="stone").set(
        body_background_fill="#fff7ec", body_background_fill_dark="#1a1714")
except Exception:
    THEME = None

CSS = """
.gradio-container { max-width: 780px !important; margin: 0 auto !important; }
footer { display: none !important; }
.card { border: 1px solid var(--border-color-primary); border-radius: 14px;
        padding: 14px 18px !important; background: var(--block-background-fill); gap: 10px !important; }
.card h4 { margin: 0 0 2px 0 !important; }
.nudge-row, .save-row, .search-row { align-items: flex-end !important; gap: 8px !important; }
.hint p { margin: 0 !important; opacity: 0.7; font-size: 0.85em; }
.banner p { margin: 0 !important; font-size: 1.05em; }
"""

# Gradio ke naye version mein theme/css launch() mein jaate hain, purane mein Blocks() mein
try:
    _css_in_launch = "css" in inspect.signature(gr.Blocks.launch).parameters
except Exception:
    _css_in_launch = False
_style = {"css": CSS}
if THEME is not None:
    _style["theme"] = THEME

parts = list_parts()

with gr.Blocks(title="Ramayan Lyric Finder", **({} if _css_in_launch else _style)) as demo:
    gr.Markdown("## 🎵 Ramayan Lyric Finder\nLyric daalo → gaana dhundho → clip aur SRT save karo")
    win = gr.State(None)
    srt_meta = gr.State(None)

    # ---- 1
    with gr.Column(elem_classes="card"):
        gr.Markdown("#### 1 · Lyric daalo")
        part = gr.Dropdown(choices=parts, value=parts[0] if parts else None, label="Part")
        lyric = gr.Textbox(
            lines=5, label="Lyric",
            placeholder="Book se lyric yahan paste karo.\nSubtitle ki har line alag banani ho to lyric ko alag-alag line mein rakho.")
        search_btn = gr.Button("🔍  Dhundho", variant="primary", size="lg")
        result = gr.Markdown(elem_classes="banner")

    # ---- 2
    with gr.Column(elem_classes="card"):
        gr.Markdown("#### 2 · Sun ke check karo")
        player = gr.Audio(label="Clip", type="filepath", interactive=False)
        with gr.Row(elem_classes="search-row"):
            info = gr.Markdown(scale=3)
            replay_btn = gr.Button("▶ Sunao", size="sm", scale=1, min_width=90)
        with gr.Row(elem_classes="nudge-row"):
            with gr.Column(scale=1, min_width=250):
                with gr.Row(elem_classes="nudge-row"):
                    s_minus = gr.Button("−0.5", size="sm", min_width=50)
                    start = gr.Number(label="Start (sec)", precision=1, min_width=90)
                    s_plus = gr.Button("+0.5", size="sm", min_width=50)
            with gr.Column(scale=1, min_width=250):
                with gr.Row(elem_classes="nudge-row"):
                    e_minus = gr.Button("−0.5", size="sm", min_width=50)
                    end = gr.Number(label="End (sec)", precision=1, min_width=90)
                    e_plus = gr.Button("+0.5", size="sm", min_width=50)
        gr.Markdown("Start/End ko ±0.5 sec se badlo, clip apne aap dobara ban jayega.", elem_classes="hint")

    # ---- 3
    with gr.Column(elem_classes="card"):
        gr.Markdown("#### 3 · Save karo")
        with gr.Row(elem_classes="save-row"):
            name = gr.Textbox(label="File ka naam (khaali chhodo to apne aap ban jayega)", scale=3, min_width=200)
            save_btn = gr.Button("💾  Clip + SRT save karo", variant="primary", scale=2, min_width=160)
        saved = gr.Markdown(elem_classes="banner")

    with gr.Accordion("📝 Subtitles (SRT) dekho / edit karo", open=False):
        srt_box = gr.Textbox(lines=12, label="SRT", interactive=True)
        srt_info = gr.Markdown(elem_classes="hint")
        srt_btn = gr.Button("SRT dobara banao", size="sm")
    with gr.Accordion("⚙️ Settings", open=False):
        lead_in = gr.Slider(0, 5, value=2, step=0.5, label="Gaane se pehle kitna extra (sec)")
        tail = gr.Slider(0, 5, value=1, step=0.5, label="Gaane ke baad kitna extra (sec)")

    # ---- wiring
    search_btn.click(do_search, [part, lyric, lead_in, tail],
                     [result, start, end, player, info, win, srt_box, srt_info, srt_meta]
                     ).then(do_preview, [part, start, end], [player, info]
                            ).then(do_srt, [part, lyric, start, end, win], [srt_box, srt_info, srt_meta])
    replay_btn.click(do_preview, [part, start, end], [player, info])
    srt_btn.click(do_srt, [part, lyric, start, end, win], [srt_box, srt_info, srt_meta])
    save_btn.click(do_save_all, [part, lyric, start, end, name, win, srt_box, srt_meta],
                   [saved, srt_box, srt_meta])
    for btn, field, delta in ((s_minus, start, -0.5), (s_plus, start, 0.5),
                              (e_minus, end, -0.5), (e_plus, end, 0.5)):
        btn.click(lambda v, d=delta: nudge(v, d), field, field
                  ).then(do_preview, [part, start, end], [player, info])

if __name__ == "__main__":
    demo.launch(inbrowser=True, **(_style if _css_in_launch else {}))