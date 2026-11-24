"""Tiny LLM client. Groq (free key) first, Pollinations keyless text as fallback for text-only work."""
import base64
import json
import os

import requests


def cfg(name, default=None):
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
POLL_URL = "https://text.pollinations.ai/openai"


_MODELS = {}


def groq_models(key):
    if "ids" not in _MODELS:
        r = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        _MODELS["ids"] = [m["id"] for m in r.json().get("data", [])]
    return _MODELS["ids"]


TEXT_PREF = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]


def pick_text(key):
    ids = groq_models(key)
    for m in TEXT_PREF:
        if m in ids:
            return m
    bad = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "distil-whisper")
    ok = [i for i in ids if not any(b in i.lower() for b in bad)]
    if not ok:
        raise RuntimeError("No chat model available on this Groq key")
    return ok[0]


def vision_candidates(key):
    """Groq models that might accept images: known vision names first, then multimodal-capable families (qwen)."""
    ids = groq_models(key)
    bad = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "safeguard", "allam")
    ids = [i for i in ids if not any(b in i.lower() for b in bad)]
    first = [i for i in ids if any(t in i.lower() for t in ("scout", "maverick", "vision", "llama-4", "-vl", "vl-"))]
    second = [i for i in ids if "qwen" in i.lower() and i not in first]
    return first + second


def _post(url, headers, payload, timeout=60):
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def chat(system, user, json_mode=False, temperature=0.6):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    errors = []
    key = cfg("GROQ_API_KEY")
    if key:
        try:
            p = {"model": cfg("GROQ_TEXT_MODEL") or pick_text(key), "messages": msgs, "temperature": temperature}
            if json_mode:
                p["response_format"] = {"type": "json_object"}
            return _post(GROQ_URL, {"Authorization": f"Bearer {key}"}, p)
        except Exception as e:
            errors.append(f"Groq: {e}")
    try:
        return _post(POLL_URL, {}, {"model": "openai", "messages": msgs, "temperature": temperature,
                                    **({"response_format": {"type": "json_object"}} if json_mode else {})})
    except Exception as e:
        errors.append(f"Pollinations text: {e}")
    raise RuntimeError("No language model reachable. " + " | ".join(errors))


def vision(system, user, image_bytes, mime="image/jpeg"):
    b64 = base64.b64encode(image_bytes).decode()
    content = [{"type": "text", "text": user}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    errors = []
    key = cfg("GROQ_API_KEY")
    if key:
        try:
            models = [cfg("GROQ_VISION_MODEL")] if cfg("GROQ_VISION_MODEL") else vision_candidates(key)
        except Exception as e:
            models, errors = [], [f"Groq model list: {e}"]
        for m in models[:3]:
            try:
                return _post(GROQ_URL, {"Authorization": f"Bearer {key}"},
                             {"model": m, "messages": msgs, "temperature": 0.4}, timeout=90)
            except Exception as e:
                errors.append(f"Groq {m}: {str(e)[:90]}")
    try:  # free keyless fallback
        return _post(POLL_URL, {}, {"model": "openai", "messages": msgs, "temperature": 0.4}, timeout=90)
    except Exception as e:
        errors.append(f"Pollinations vision: {str(e)[:90]}")
    raise RuntimeError("No vision model worked. " + " | ".join(errors))


def parse_json(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(s[s.find("{"): s.rfind("}") + 1])
