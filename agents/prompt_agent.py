"""Prompt Agent (director): turns a short idea into a detailed photorealistic prompt."""
from agents import llm

IMAGE_SYS = (
    "You are a professional photographer and prompt engineer. Rewrite the user's idea as ONE detailed prompt for a "
    "photorealistic image model. Include: subject, setting, lighting, lens/camera (e.g. 35mm, f/1.8), composition, "
    "color mood, natural imperfections (skin texture, film grain), time of day. No names of real people. "
    "Max 110 words. Output only the prompt."
)
VIDEO_SYS = (
    "You are a cinematographer. Rewrite the idea as ONE prompt for a text-to-video model: describe the subject, "
    "ONE clear camera move (slow dolly, pan, handheld), motion of the subject, lighting, realistic physics, "
    "cinematic 24fps look. 60-90 words. No names of real people. Output only the prompt."
)
VISION_SYS = (
    "You describe images so they can be recreated by an AI image generator. Describe the scene, composition, "
    "lighting, colors, camera angle, style and mood in one detailed paragraph. Never identify or name real people; "
    "describe them generically (e.g. 'a middle-aged man in a grey suit')."
)


class PromptAgent:
    name = "PromptAgent"

    def improve(self, idea, kind="image", style=""):
        sys = IMAGE_SYS if kind == "image" else VIDEO_SYS
        extra = f"\nStyle direction: {style}" if style and style != "Photorealistic (default)" else ""
        return llm.chat(sys, idea + extra)


class VisionAgent:
    name = "VisionAgent"

    def describe(self, image_bytes, mime="image/jpeg", goal="image"):
        ask = "Write a prompt that recreates this image." if goal == "image" else \
              "Write a prompt for a short video that starts from this scene and adds natural, subtle motion."
        return llm.vision(VISION_SYS, ask, image_bytes, mime)


ACTION_SYS = (
    "You write prompts for an image-to-{kind} model. You get (1) a description of the uploaded picture and (2) the action "
    "the user wants. Write ONE prompt that keeps the picture's subject, setting, lighting and style unchanged and adds only "
    "the requested action with natural, physically realistic motion{cam}. Never name real people. 50-90 words. Output only the prompt."
)


def action_prompt(description, action, kind="video"):
    from agents import llm
    sys = ACTION_SYS.format(kind=kind, cam=" and one subtle camera move" if kind == "video" else "")
    return llm.chat(sys, f"Picture: {description}\nAction wanted: {action}")
