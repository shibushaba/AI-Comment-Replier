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

# ------------------ Global Minimal Styling ------------------

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}

/* Headings */
h2, h3, h4 {
    color: #111827;
}

/* Buttons */
.stButton > button {
    background-color: #111827;
    color: white;
    border-radius: 6px;
    padding: 0.55rem 1rem;
    border: none;
}

.stButton > button:hover {
    background-color: #1f2937;
}

/* Inputs */
textarea, select {
    border-radius: 6px !important;
}

/* Code blocks */
pre {
    background-color: #f9fafb !important;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
}

/* Containers */
[data-testid="stContainer"] {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------

st.markdown("""
<h2>AI Comment Reply Helper</h2>
<p style="color:#6b7280; margin-top:4px;">
Generate platform-aware, tone-perfect replies for creators and professionals
</p>
<hr style="margin: 1.5rem 0;">
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
        "No Gemini text-generation models available for this API key.\n\n"
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
    st.markdown("#### Input")

    comment = st.text_area(
        "Comment",
        placeholder="This video was really helpful, thanks!",
        height=120
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

    generate = st.button("Generate Replies", use_container_width=True)

# ------------------ Reply Card ------------------

def reply_card(title, text):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(
            f"<div style='color:#374151; margin-top:8px;'>{text}</div>",
            unsafe_allow_html=True
        )
        st.code(text, language="text")

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
