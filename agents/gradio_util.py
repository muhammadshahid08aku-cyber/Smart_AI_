"""Call any public Hugging Face Space through gradio_client without hard-coding its parameter list."""
import os


def _choose(eps, has_image):
    """Pick the endpoint that fits: image-to-video when an image is given, text-to-video otherwise."""
    def score(n):
        low, sc = n.lower(), 0
        pnames = " ".join(p["parameter_name"].lower() for p in eps[n]["parameters"])
        wants = ("image", "i2v") if has_image else ("text", "t2v", "prompt")
        if any(w in low for w in wants):
            sc += 3
        if has_image and "image" in pnames:
            sc += 2
        if not has_image and "image" not in pnames:
            sc += 2
        if any(w in low for w in ("generate", "video", "infer", "t2v", "i2v", "run")):
            sc += 1
        if any(w in low for w in ("example", "clear", "reset", "update", "toggle", "load")):
            sc -= 5
        return sc
    return max(eps, key=score)


def _mode_value(p, has_image):
    """Spaces like LTX have one endpoint with a mode dropdown: choose image-to-video or text-to-video."""
    want = "image" if has_image else "text"
    t = p.get("type") or {}
    choices = (t.get("enum") if isinstance(t, dict) else None) or []
    for c in choices:
        if want in str(c).lower():
            return c
    return "image-to-video" if has_image else "text-to-video"


def call_space(space, prompt, api_name=None, image_path=None, extra=None, token=None):
    from gradio_client import Client, handle_file
    client = Client(space, token=token or os.environ.get("HF_TOKEN") or None)
    info = client.view_api(return_format="dict", print_info=False)
    eps = info["named_endpoints"]
    name = api_name if api_name in eps else _choose(eps, bool(image_path))
    args = {}
    for p in eps[name]["parameters"]:
        n = p["parameter_name"]
        low = n.lower()
        if extra and n in extra:
            args[n] = extra[n]
        elif "mode" in low and "prompt" not in low:
            args[n] = _mode_value(p, bool(image_path))
        elif "negative" in low:
            args[n] = "blurry, low quality, distorted, watermark, text"
        elif "prompt" in low:
            args[n] = prompt
        elif "image" in low and image_path:
            args[n] = handle_file(image_path)
        elif p.get("parameter_has_default"):
            args[n] = p["parameter_default"]
        elif not p.get("parameter_has_default") and "image" in low:
            raise RuntimeError(f"{space} needs an input image")
    try:
        return client.predict(api_name=name, **args)
    except Exception as e:
        sent = {k: ("<file>" if "FileData" in repr(v) or "handle_file" in repr(v) or (isinstance(v, dict) and "path" in v) else str(v)[:30])
                for k, v in args.items()}
        raise RuntimeError(f"{str(e)[:150]} [endpoint {name}; sent {sent}; image given: {bool(image_path)}]") from e
