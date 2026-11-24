"""Video Agent: free video generation through public Hugging Face Spaces (ZeroGPU). Slow, quota-limited."""
import shutil
import tempfile
from pathlib import Path

from agents import llm
from agents.gradio_util import call_space

DEFAULT_SPACES = ["Lightricks/ltx-video-distilled", "Wan-AI/Wan2.2-5B"]


def _find_video(out):
    if isinstance(out, (list, tuple)):
        for o in out:
            v = _find_video(o)
            if v:
                return v
    elif isinstance(out, dict):
        return out.get("video") and _find_video(out["video"]) or out.get("path") or out.get("url")
    elif isinstance(out, str) and out.lower().endswith((".mp4", ".webm", ".mov")):
        return out
    return None


class VideoAgent:
    name = "VideoAgent"

    def generate(self, prompt, image_bytes=None, log=None):
        spaces = [s.strip() for s in (llm.cfg("VIDEO_SPACES") or "").split(",") if s.strip()] or DEFAULT_SPACES
        img_path = None
        if image_bytes:
            img_path = str(Path(tempfile.mkdtemp()) / "in.jpg")
            Path(img_path).write_bytes(image_bytes)
        errors = []
        for sp in spaces:
            try:
                if log:
                    log(f"Asking {sp} (can take 1-5 minutes)...")
                out = call_space(sp, prompt, image_path=img_path, token=llm.cfg("HF_TOKEN"))
                vid = _find_video(out)
                if not vid:
                    raise RuntimeError("no video in the response")
                dst = Path(tempfile.mkdtemp()) / "video.mp4"
                shutil.copy(vid, dst)
                return dst.read_bytes(), sp
            except Exception as e:
                errors.append(f"{sp}: {str(e)[:200]}")
                if log:
                    log(f"{sp} failed: {str(e)[:120]}")
        raise RuntimeError("All free video Spaces failed (they are often busy or out of daily quota).\n" + "\n".join(errors))
