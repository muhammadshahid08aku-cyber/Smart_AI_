"""Prompt-to-Image and Prompt-to-Video studio with free tools and a team of agents."""
import mimetypes

import streamlit as st

from agents import llm
from agents.image_agent import EditAgent, ImageAgent
from agents.prompt_agent import PromptAgent, VisionAgent, action_prompt
from agents.safety_agent import SafetyAgent
from agents.video_agent import VideoAgent

st.set_page_config(page_title="AI Image & Video Studio", page_icon="🎬", layout="wide")
safety, prompter, vision_a, image_a, video_a = SafetyAgent(), PromptAgent(), VisionAgent(), ImageAgent(), VideoAgent()
edit_a = EditAgent()
STYLES = ["Photorealistic (default)", "Cinematic film still", "Studio product photo", "Documentary / street photo",
          "Macro nature photo", "Architectural photo"]
SIZES = {"Square 768x768": (768, 768), "Portrait 576x768": (576, 768), "Landscape 768x576": (768, 576),
         "Wide 1024x576": (1024, 576)}

st.title("🎬 AI Image & Video Studio")
st.caption("Free tools only. A team of agents: Safety → Prompt Director → Vision → Image / Video generators.")

with st.sidebar:
    st.subheader("Setup status")
    st.write("Groq key (prompts + image-to-prompt):", "✅" if llm.cfg("GROQ_API_KEY") else "❌ add GROQ_API_KEY (free)")
    st.write("Hugging Face token (more free quota):", "✅" if llm.cfg("HF_TOKEN") else "optional")
    st.info("Free services have limits: small images, queues, daily quotas. Results are labelled as AI-generated.")


def run_safety(text):
    ok, why = safety.check(text)
    if not ok:
        st.error(why + " Please change the request.")
    return ok


def prompt_box(key, kind, seed_text=""):
    idea = st.text_area("Your idea (short is fine)", value=seed_text, key=f"idea_{key}", height=100,
                        placeholder="a fisherman mending nets at sunrise on a misty lake")
    style = st.selectbox("Style", STYLES, key=f"style_{key}")
    auto = st.toggle("Let the Prompt Director improve it", True, key=f"auto_{key}")
    return idea, style, auto


def final_prompt(idea, style, auto, kind):
    if not auto:
        return idea
    with st.spinner("Prompt Director is writing a detailed prompt..."):
        try:
            p = prompter.improve(idea, kind, style)
        except Exception as e:
            st.warning(f"Prompt Director unavailable, using your text as is. ({str(e)[:100]})")
            return idea
    st.session_state[f"last_prompt_{kind}"] = p
    return p


tab_img, tab_act, tab_describe, tab_vid = st.tabs(["🖼️ Text → Image", "🎯 Picture + Action", "🔍 Image → Prompt", "🎞️ Text → Video"])

# ---------------------------------------------------------------- image
with tab_img:
    idea, style, auto = prompt_box("img", "image", st.session_state.pop("send_to_img", ""))
    size = st.selectbox("Size", list(SIZES))
    seed = st.number_input("Seed (0 = random)", 0, 999999, 0)
    if st.button("Generate image", type="primary") and idea.strip():
        if run_safety(idea):
            p = final_prompt(idea, style, auto, "image")
            if run_safety(p):
                st.code(p, language=None)
                box = st.status("Image agent working...", expanded=True)
                try:
                    data, prov = image_a.generate(p, *SIZES[size], seed or None, log=box.write)
                    box.update(label=f"Done with {prov}", state="complete")
                    st.image(data, caption="AI-generated image")
                    st.download_button("Download", data, "image.jpg", "image/jpeg")
                except Exception as e:
                    box.update(label="Failed", state="error")
                    st.error(str(e))


# ---------------------------------------------------------------- picture + action
with tab_act:
    st.write("Upload a picture, then say what should happen. Example: *she turns and walks toward the camera, wind in her hair*.")
    pic = st.file_uploader("Your picture", type=["jpg", "jpeg", "png", "webp"], key="actpic")
    action = st.text_area("Action prompt", key="act_text", height=90)
    out_kind = st.radio("Result", ["Video (animate it)", "Picture (edit it)"], horizontal=True)
    if pic:
        st.image(pic, width=300)
        st.warning("Use only pictures you own or have permission to use. Realistic edits or videos of real people who have "
                   "not agreed can be used to deceive or harm, and are not allowed here.")
        ok_own = st.checkbox("I own this picture or have permission, and it does not show a real person who has not agreed")
        if st.button("Go", type="primary") and action.strip():
            kind = "video" if out_kind.startswith("Video") else "image"
            if not ok_own:
                st.error("Please confirm permission first.")
            elif run_safety(action):
                with st.spinner("Vision agent is looking, Prompt Director is planning..."):
                    try:
                        mime = pic.type or "image/jpeg"
                        desc = vision_a.describe(pic.getvalue(), mime, kind)
                        p = action_prompt(desc, action, kind)
                    except Exception as e:
                        st.warning(f"Using your action text directly ({str(e)[:100]})")
                        p = action
                if run_safety(p):
                    st.code(p, language=None)
                    box = st.status("Working (free GPUs queue, be patient)...", expanded=True)
                    try:
                        if kind == "video":
                            data, sp = video_a.generate(p, pic.getvalue(), log=box.write)
                            box.update(label=f"Done with {sp}", state="complete")
                            st.video(data)
                            st.download_button("Download MP4", data, "video.mp4", "video/mp4")
                        else:
                            data, sp = edit_a.edit(pic.getvalue(), action, log=box.write)
                            box.update(label=f"Done with {sp}", state="complete")
                            st.image(data, caption="AI-edited image")
                            st.download_button("Download", data, "edited.jpg", "image/jpeg")
                    except Exception as e:
                        box.update(label="Failed", state="error")
                        st.error(str(e))

# ---------------------------------------------------------------- image to prompt
with tab_describe:
    up = st.file_uploader("Upload a picture to get a prompt that recreates it", type=["jpg", "jpeg", "png", "webp"])
    goal = st.radio("Prompt for", ["image", "video"], horizontal=True)
    if up:
        st.image(up, width=320)
        if st.button("Create prompt from this picture", type="primary"):
            with st.spinner("Vision agent is looking..."):
                try:
                    mime = up.type or mimetypes.guess_type(up.name)[0] or "image/jpeg"
                    st.session_state["desc"] = vision_a.describe(up.getvalue(), mime, goal)
                except Exception as e:
                    st.error(str(e))
    if st.session_state.get("desc"):
        txt = st.text_area("Prompt (edit if you like)", st.session_state["desc"], height=180)
        st.caption("Copy it into the Image or Video tab. Uploaded photos are only described; people are never identified.")

# ---------------------------------------------------------------- video
with tab_vid:
    idea, style, auto = prompt_box("vid", "video", "")
    start = st.file_uploader("Optional: start from a picture (image → video)", type=["jpg", "jpeg", "png", "webp"], key="vidimg")
    if start:
        st.warning("Only animate pictures you own or have permission to use. Photos of real people are not allowed.")
        own = st.checkbox("I own this picture / have permission, and it is not a photo of a real person I could harm or mislead with")
    else:
        own = True
    if st.button("Generate video", type="primary") and idea.strip():
        if not own:
            st.error("Please confirm the permission checkbox.")
        elif run_safety(idea):
            p = final_prompt(idea, style, auto, "video")
            if run_safety(p):
                st.code(p, language=None)
                box = st.status("Video agent working (free GPUs queue, be patient)...", expanded=True)
                try:
                    data, sp = video_a.generate(p, start.getvalue() if start else None, log=box.write)
                    box.update(label=f"Done with {sp}", state="complete")
                    st.video(data)
                    st.download_button("Download MP4", data, "video.mp4", "video/mp4")
                except Exception as e:
                    box.update(label="Failed", state="error")
                    st.error(str(e))
