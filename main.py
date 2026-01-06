# ================================
# AI Comment Reply Helper
# LangChain Core + Gemini SDK (AUTO MODEL SAFE)
# ================================

import os
import uuid
import streamlit as st
from pydantic import BaseModel
import google.generativeai as genai
import chromadb

# LangChain CORE
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

# ------------------ Page Config ------------------

st.set_page_config(
    page_title="AI Comment Reply Helper",
    layout="centered"
)

# ------------------ Global Styling ------------------

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    padding: 3rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2.5rem;
}
.hero h1 {
    font-size: 2.4rem;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #cbd5f5;
    max-width: 650px;
    font-size: 1.05rem;
}

/* Buttons */
.stButton > button {
    background-color: #0f172a;
    color: white;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    border: none;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: #020617;
}

/* Inputs */
textarea, select {
    border-radius: 8px !important;
}

/* Containers */
[data-testid="stContainer"] {
    border-radius: 12px;
}

/* Code blocks */
pre {
    background-color: #f8fafc !important;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Hero Section ------------------

st.markdown("""
<div class="hero">
    <h1>AI Comment Reply Helper</h1>
    <p>
        Generate human-like, platform-aware replies that build trust,
        boost engagement, and save time for creators and professionals.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------ Value Props ------------------

st.markdown("""
<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:1.5rem; margin-bottom:2.5rem;">
    <div><strong>Human-like replies</strong><br><span style="color:#64748b;">No robotic tone</span></div>
    <div><strong>Platform-aware</strong><br><span style="color:#64748b;">YouTube ≠ LinkedIn</span></div>
    <div><strong>Fast & reliable</strong><br><span style="color:#64748b;">Replies in seconds</span></div>
</div>
""", unsafe_allow_html=True)

# ------------------ API KEY ------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "Gemini API key not found.\n\n"
        "Set it using:\n"
        "setx GOOGLE_API_KEY \"YOUR_API_KEY\""
    )
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ------------------ SAFE MODEL SELECTION ------------------

@st.cache_resource
def get_supported_model():
    models = genai.list_models()
    for m in models:
        if "generateContent" in m.supported_generation_methods:
            return m.name
    return None

MODEL_NAME = get_supported_model()

if not MODEL_NAME:
    st.error(
        "No Gemini text-generation models available.\n"
        "Enable Generative Language API in Google AI Studio."
    )
    st.stop()

st.caption(f"Model in use: {MODEL_NAME}")

model = genai.GenerativeModel(MODEL_NAME)

# ------------------ ChromaDB Memory ------------------

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("comment_replies")

def save_reply(comment, platform, tone, replies):
    collection.add(
        documents=[comment],
        metadatas=[{
            "platform": platform,
            "tone": tone,
            "friendly": replies.friendly_reply,
            "professional": replies.professional_reply,
            "engagement": replies.engagement_boosting_reply,
        }],
        ids=[str(uuid.uuid4())]
    )

# ------------------ Schema ------------------

class CommentReplies(BaseModel):
    friendly_reply: str
    professional_reply: str
    engagement_boosting_reply: str

parser = PydanticOutputParser(pydantic_object=CommentReplies)

# ------------------ Prompt ------------------

prompt = PromptTemplate(
    template="""
You are helping a content creator reply to comments.

Platform: {platform}
Tone: {tone}

Comment:
"{comment}"

{format_instructions}

Rules:
- Sound human and natural
- Match the platform style
- Respect the tone
- Keep replies concise
""",
    input_variables=["comment", "platform", "tone"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# ------------------ Gemini Runnable ------------------

def gemini_call(prompt_value) -> str:
    prompt_text = prompt_value.to_string()
    response = model.generate_content(prompt_text)
    return response.text.strip()

llm = RunnableLambda(gemini_call)

# ------------------ LangChain Chain ------------------

chain = prompt | llm | parser

# ------------------ Input UI ------------------

with st.container(border=True):
    st.markdown("### Write a comment")

    comment = st.text_area(
        label="",
        placeholder="Paste a comment you received on your post or video…",
        height=140
    )

    col1, col2 = st.columns(2)

    with col1:
        platform = st.selectbox(
            "Platform",
            ["YouTube", "Instagram", "LinkedIn"]
        )

    with col2:
        tone = st.selectbox(
            "Tone",
            ["Casual", "Friendly", "Professional", "Formal"],
            index=1
        )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    generate = st.button("Generate replies", use_container_width=True)

# ------------------ Reply Card ------------------

def reply_card(title, text):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.markdown(
            f"""
            <div style="
                background:#f8fafc;
                padding:1rem;
                border-radius:10px;
                border:1px solid #e5e7eb;
                font-size:0.95rem;
                color:#0f172a;
            ">
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------ Output ------------------

if generate and comment.strip():
    with st.spinner("Generating replies..."):
        replies = chain.invoke({
            "comment": comment,
            "platform": platform,
            "tone": tone
        })
        save_reply(comment, platform, tone, replies)

    st.markdown("### Generated Replies")

    reply_card("Friendly Reply", replies.friendly_reply)
    reply_card("Professional Reply", replies.professional_reply)
    reply_card("Engagement-Boosting Reply", replies.engagement_boosting_reply)

elif generate:
    st.warning("Please enter a comment to continue.")
