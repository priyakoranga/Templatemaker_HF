
import streamlit as st
import requests
import os
import re

# ─── CONFIG ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# ─── HUGGING FACE CALL ────────────────────────────────────────────────────────
def call_huggingface(prompt_text: str) -> str | None:
    if not HF_TOKEN:
        return None
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt_text,
        "parameters": {"max_new_tokens": 600, "temperature": 0.7, "do_sample": True},
    }
    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            raw = resp.json()
            # Extract only the generated text after our prompt
            if isinstance(raw, list) and raw:
                text = raw[0].get("generated_text", "")
                if "TEMPLATES:" in text:
                    text = text.split("TEMPLATES:")[-1].strip()
                return text
    except Exception:
        pass
    return None

# ─── STRONG RULE-BASED FALLBACK ───────────────────────────────────────────────
TEMPLATE_STYLES = [
    {
        "label": "🎯 Expert Mode",
        "icon": "🎯",
        "color": "#1a73e8",
        "build": lambda topic: (
            f"Act as a world-class expert with 20+ years of experience in {topic}. "
            f"Provide a comprehensive, authoritative explanation of {topic} that covers: "
            f"(1) the core principles and why they matter, "
            f"(2) common misconceptions and how to avoid them, "
            f"(3) advanced insights that most people overlook, "
            f"(4) real-world examples and case studies. "
            f"Write for a professional audience that wants depth, not just surface-level information."
        ),
    },
    {
        "label": "📋 Step-by-Step Guide",
        "icon": "📋",
        "color": "#0f9d58",
        "build": lambda topic: (
            f"Provide a detailed, practical step-by-step guide on {topic}. "
            f"Structure your response as follows: "
            f"Step 1: Start with the absolute basics — what do I need before I begin? "
            f"Step 2–5: Walk me through each phase clearly, with specific actions at each stage. "
            f"Step 6: What are the most common mistakes and how do I avoid them? "
            f"Step 7: How do I know if I am doing it correctly? Include checkpoints. "
            f"Use simple language, numbered lists, and concrete examples throughout."
        ),
    },
    {
        "label": "🧑‍🎓 Beginner Friendly",
        "icon": "🧑‍🎓",
        "color": "#f4511e",
        "build": lambda topic: (
            f"Explain {topic} to a complete beginner who has zero prior knowledge. "
            f"Use simple everyday language — no jargon without explanation. "
            f"Start with a one-sentence definition a 12-year-old could understand. "
            f"Then use a real-life analogy to make it relatable. "
            f"Cover the 3 most important things a beginner absolutely must know about {topic}. "
            f"End with 2–3 beginner-friendly action steps they can take today."
        ),
    },
    {
        "label": "📊 Structured Analysis",
        "icon": "📊",
        "color": "#ab47bc",
        "build": lambda topic: (
            f"Analyze {topic} using a structured framework. "
            f"Format your response with clear headings: "
            f"**Overview** — What is {topic} in 2–3 sentences? "
            f"**Key Benefits** — List the top 5 advantages with a brief explanation for each. "
            f"**Challenges & Risks** — What are the main pitfalls? Be specific. "
            f"**Best Practices** — What do top performers do differently? "
            f"**Tools & Resources** — Name 3–5 specific tools, books, or methods. "
            f"**Action Plan** — Give a 30-day roadmap for someone starting today."
        ),
    },
    {
        "label": "💡 Creative Exploration",
        "icon": "💡",
        "color": "#ff8f00",
        "build": lambda topic: (
            f"Explore {topic} from unexpected and creative angles. "
            f"Go beyond the obvious — I want fresh perspectives that most articles miss. "
            f"Include: "
            f"(1) A surprising or counterintuitive fact about {topic} that challenges conventional wisdom. "
            f"(2) How {topic} connects to an unrelated field in an interesting way. "
            f"(3) A thought experiment: what would the world look like if {topic} did not exist? "
            f"(4) What experts argue about regarding {topic} — show me the debate. "
            f"Be imaginative, thought-provoking, and intellectually stimulating."
        ),
    },
]

def generate_fallback_templates(topic: str) -> list[dict]:
    topic = topic.strip().lower()
    return [
        {"label": s["label"], "icon": s["icon"], "color": s["color"], "text": s["build"](topic)}
        for s in TEMPLATE_STYLES
    ]

# ─── PARSE AI OUTPUT INTO TEMPLATES ──────────────────────────────────────────
def parse_ai_templates(raw: str, topic: str) -> list[dict]:
    """Try to extract numbered templates from AI output; fall back if malformed."""
    blocks = re.split(r"\n?\d+[\.\)]\s*", raw)
    blocks = [b.strip() for b in blocks if len(b.strip()) > 60]
    if len(blocks) < 3:
        return []   # tell caller to use fallback
    styles = TEMPLATE_STYLES
    results = []
    for i, block in enumerate(blocks[:5]):
        style = styles[i % len(styles)]
        results.append({
            "label": style["label"],
            "icon": style["icon"],
            "color": style["color"],
            "text": block,
        })
    return results

# ─── MAIN GENERATION FUNCTION ─────────────────────────────────────────────────
def generate_templates(topic: str) -> tuple[list[dict], str]:
    """Returns (templates, source) where source is 'ai' or 'fallback'."""
    meta_prompt = (
        f"You are a prompt engineering expert. The user wants help with the topic: \"{topic}\". "
        f"Generate exactly 5 distinct, high-quality prompt templates that a user can paste into ChatGPT. "
        f"Each template must be: specific, detailed, actionable, and clearly different from the others. "
        f"Use different styles: expert mode, step-by-step, beginner-friendly, analytical, and creative. "
        f"Number each template 1. 2. 3. 4. 5. and write at least 3 sentences per template. "
        f"TEMPLATES:"
    )
    raw = call_huggingface(meta_prompt)
    if raw:
        parsed = parse_ai_templates(raw, topic)
        if parsed:
            return parsed, "ai"
    # fallback
    return generate_fallback_templates(topic), "fallback"

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prompt Template Generator", page_icon="✨", layout="centered")

st.markdown("""
<style>
.template-card {
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
    border-left: 4px solid;
    background: #f8f9fa;
}
.template-label {
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 6px;
}
.template-text {
    font-size: 14px;
    line-height: 1.65;
    color: #2d2d2d;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

st.title("✨ Prompt Template Generator")
st.caption("Turn any basic topic into 5 high-quality, ready-to-use AI prompts.")

st.markdown("---")

topic = st.text_input(
    "Enter your topic or idea:",
    placeholder="e.g.  success,  machine learning,  healthy habits,  public speaking",
    max_chars=120,
)

col1, col2 = st.columns([2, 1])
with col1:
    generate_btn = st.button("🚀 Generate Templates", use_container_width=True, type="primary")
with col2:
    st.caption("Free · No login needed")

if generate_btn:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Generating high-quality prompt templates for you…"):
            templates, source = generate_templates(topic.strip())

        if source == "ai":
            st.success("✅ Templates generated using AI (Hugging Face)")
        else:
            st.info("💡 Templates generated using smart rule-based system (AI unavailable or token not set)")

        st.markdown(f"### 5 Prompt Templates for: **{topic.strip()}**")

        for i, tmpl in enumerate(templates, 1):
            with st.expander(f"{tmpl['icon']} Template {i} — {tmpl['label']}", expanded=True):
                st.markdown(
                    f'<div class="template-card" style="border-left-color:{tmpl["color"]}">'
                    f'<div class="template-label" style="color:{tmpl["color"]}">{tmpl["label"]}</div>'
                    f'<div class="template-text">{tmpl["text"]}</div></div>',
                    unsafe_allow_html=True,
                )
                st.code(tmpl["text"], language=None)   # gives a one-click copy button ✅

        st.markdown("---")
        st.caption("💡 Tip: paste any template directly into ChatGPT, Claude, or Gemini for best results.")

st.markdown("---")
with st.expander("⚙️ Setup: Connect your Hugging Face API key"):
    st.markdown("""
1. Get a free key at [huggingface.co](https://huggingface.co) → Settings → Access Tokens
2. In the Colab cell below the app, set `os.environ["HF_TOKEN"] = "hf_your_key_here"`
3. Re-run the launch cell — the app will now use real AI generation
""")
