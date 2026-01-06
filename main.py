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
    page_icon="💬",
    layout="centered"
)

st.title("💬 AI Comment Reply Helper")
st.caption("Platform-aware • Tone-controlled • Creator-friendly")

# ------------------ API KEY ------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "❌ Gemini API key not found.\n\n"
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
        "❌ No Gemini text-generation models available for this API key.\n\n"
        "Your Google project does not have access to Generative Language models.\n"
        "Enable **Generative Language API** in Google AI Studio or switch provider."
    )
    st.stop()

st.success(f"✅ Using Gemini model: `{MODEL_NAME}`")

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

# ------------------ UI ------------------

with st.container(border=True):
    comment = st.text_area(
        "Paste the comment you received",
        placeholder="This video was really helpful, thanks!"
    )

    col1, col2 = st.columns(2)

    with col1:
        platform = st.selectbox(
            "Platform",
            ["YouTube", "Instagram", "LinkedIn"]
        )

    with col2:
        tone = st.select_slider(
            "Tone",
            ["Casual", "Friendly", "Professional", "Formal"],
            value="Friendly"
        )

generate = st.button("✨ Generate Replies", use_container_width=True)

def reply_card(title, text):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(text)
        st.code(text)
        st.button("📋 Copy", key=str(uuid.uuid4()))

# ------------------ Output ------------------

if generate and comment.strip():
    with st.spinner("Crafting replies..."):
        replies = chain.invoke({
            "comment": comment,
            "platform": platform,
            "tone": tone
        })
        save_reply(comment, platform, tone, replies)

    st.success("Replies ready 👇")

    reply_card("😊 Friendly Reply", replies.friendly_reply)
    reply_card("💼 Professional Reply", replies.professional_reply)
    reply_card("🚀 Engagement-Boosting Reply", replies.engagement_boosting_reply)

elif generate:
    st.warning("Please enter a comment first.")
