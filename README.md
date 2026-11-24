# AI Image & Video Studio (free tools + agents)

Agents: Safety -> Prompt Director -> Vision (image-to-prompt) -> Image agent / Video agent.
Image providers tried in order: Pollinations (no key) -> Hugging Face FLUX Space -> Stable Horde.
Video: public Hugging Face Spaces (LTX-Video, Wan). Prompt writing and image-to-prompt: free Groq key.

## Run
    pip install -r requirements.txt
    GROQ_API_KEY=gsk_... streamlit run app.py

## Deploy: GitHub -> share.streamlit.io -> Secrets (see .streamlit/secrets.toml.example)

## Honest limits of "free"
- Keyless Pollinations images are small (up to 768px), may carry a logo, and allow ~1 request per 45 s.
- Free video Spaces queue, hit daily GPU quotas and change; set VIDEO_SPACES to any Space you like.
- For truly HD realistic output, run FLUX/SDXL and LTX/Wan on a free Colab GPU instead.
- Safety: no real named people, no sexual content of real people or minors, no forged documents or fake-news footage.
  Label generated media as AI-generated.
