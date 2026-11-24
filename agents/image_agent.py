"""Image Agent: tries free providers in order until one works."""
import base64
import os
import tempfile
import time
import urllib.parse

import requests

from agents import llm
from agents.gradio_util import call_space


def _pollinations(prompt, w, h, seed):
    q = urllib.parse.quote(prompt[:1500])
    url = f"https://image.pollinations.ai/prompt/{q}?width={w}&height={h}&seed={seed}&nologo=true&model=flux&enhance=false"
    key = llm.cfg("POLLINATIONS_KEY")
    r = requests.get(url, headers={"Authorization": f"Bearer {key}"} if key else {}, timeout=120)
    if r.status_code == 402 or r.status_code == 429:
        raise RuntimeError(f"rate limited (retry after {r.headers.get('Retry-After', '45')}s)")
    r.raise_for_status()
    if "image" not in r.headers.get("content-type", ""):
        raise RuntimeError("not an image response")
    return r.content


def _hf_space(prompt, w, h, seed):
    space = llm.cfg("IMAGE_SPACE", "black-forest-labs/FLUX.1-schnell")
    out = call_space(space, prompt, extra={"width": w, "height": h, "seed": seed, "randomize_seed": False},
                     token=llm.cfg("HF_TOKEN"))
    path = out[0] if isinstance(out, (tuple, list)) else out
    if isinstance(path, dict):
        path = path.get("path") or path.get("url")
    with open(path, "rb") as f:
        return f.read()


def _horde(prompt, w, h, seed):
    hdr = {"apikey": llm.cfg("HORDE_KEY", "0000000000"), "Client-Agent": "media-agents:1:anon"}
    w, h = min(w, 768) // 64 * 64, min(h, 768) // 64 * 64
    body = {"prompt": prompt, "params": {"width": w, "height": h, "steps": 25, "n": 1, "sampler_name": "k_euler",
                                         "seed": str(seed)}, "nsfw": False, "r2": False, "slow_workers": True}
    r = requests.post("https://aihorde.net/api/v2/generate/async", json=body, headers=hdr, timeout=30)
    r.raise_for_status()
    rid = r.json()["id"]
    for _ in range(90):  # up to ~6 minutes
        time.sleep(4)
        st = requests.get(f"https://aihorde.net/api/v2/generate/check/{rid}", timeout=30).json()
        if st.get("done"):
            res = requests.get(f"https://aihorde.net/api/v2/generate/status/{rid}", timeout=60).json()
            img = res["generations"][0]["img"]
            return base64.b64decode(img) if not img.startswith("http") else requests.get(img, timeout=60).content
        if st.get("faulted"):
            break
    raise RuntimeError("Stable Horde queue timed out")


PROVIDERS = [("Pollinations (free, no key)", _pollinations),
             ("Hugging Face FLUX Space (free)", _hf_space),
             ("Stable Horde (free, queued)", _horde)]


class ImageAgent:
    name = "ImageAgent"

    def generate(self, prompt, w=768, h=768, seed=None, log=None):
        seed = seed if seed is not None else int(time.time()) % 1_000_000
        errors = []
        for label, fn in PROVIDERS:
            try:
                if log:
                    log(f"Trying {label}...")
                return fn(prompt, w, h, seed), label
            except Exception as e:
                errors.append(f"{label}: {str(e)[:160]}")
                if log:
                    log(f"{label} failed: {str(e)[:120]}")
        raise RuntimeError("All free image providers failed.\n" + "\n".join(errors))


class EditAgent:
    """Uploaded picture + instruction ("add snow", "make it night") -> edited picture, via a free HF Space."""
    name = "EditAgent"

    def edit(self, image_bytes, instruction, log=None):
        space = llm.cfg("EDIT_SPACE", "black-forest-labs/FLUX.1-Kontext-Dev")
        path = os.path.join(tempfile.mkdtemp(), "in.jpg")
        with open(path, "wb") as f:
            f.write(image_bytes)
        if log:
            log(f"Asking {space}...")
        out = call_space(space, instruction, image_path=path, token=llm.cfg("HF_TOKEN"))
        res = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(res, dict):
            res = res.get("path") or res.get("url")
        with open(res, "rb") as f:
            return f.read(), space
