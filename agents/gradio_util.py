"""Call any public Hugging Face Space through gradio_client without hard-coding its parameter list."""
import os


def call_space(space, prompt, api_name=None, image_path=None, extra=None, token=None):
    from gradio_client import Client, handle_file
    client = Client(space, token=token or os.environ.get("HF_TOKEN") or None)
    info = client.view_api(return_format="dict", print_info=False)
    eps = info["named_endpoints"]
    name = api_name if api_name in eps else next(iter(eps))
    args = {}
    for p in eps[name]["parameters"]:
        n = p["parameter_name"]
        low = n.lower()
        if extra and n in extra:
            args[n] = extra[n]
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
    out = client.predict(api_name=name, **args)
    return out
