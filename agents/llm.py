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
            p = {"model": cfg("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile"), "messages": msgs, "temperature": temperature}
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
    key = cfg("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Image-to-prompt needs a free Groq key (console.groq.com/keys). Add GROQ_API_KEY.")
    b64 = base64.b64encode(image_bytes).decode()
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user},
                                         {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]
    return _post(GROQ_URL, {"Authorization": f"Bearer {key}"},
                 {"model": cfg("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"), "messages": msgs,
                  "temperature": 0.4}, timeout=90)


def parse_json(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(s[s.find("{"): s.rfind("}") + 1])
