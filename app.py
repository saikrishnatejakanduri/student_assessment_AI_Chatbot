import json
import random
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Student Assistant Chatbot", page_icon="🎓", layout="centered")
DATASET_PATH = Path(__file__).parent / "intents.json"

@st.cache_data
def load_intents():
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("intents", []) if isinstance(data, dict) else data

if not DATASET_PATH.exists():
    st.error("intents.json was not found. Keep it in the same folder as app.py.")
    st.stop()
try:
    intents = load_intents()
except Exception as exc:
    st.error(f"Could not read intents.json: {exc}")
    st.stop()

training_patterns = []
pattern_tags = []
for intent in intents:
    for pattern in intent.get("patterns", []):
        training_patterns.append(pattern)
        pattern_tags.append(intent.get("tag", "fallback"))

if training_patterns:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")
    pattern_matrix = vectorizer.fit_transform(training_patterns)
else:
    vectorizer, pattern_matrix = None, None

intent_by_tag = {intent.get("tag", "fallback"): intent for intent in intents}
FALLBACK_TAGS = {"fallback", "unknown", "default"}

# Small keyword-based guidance helps users ask a more useful follow-up.
def helpful_followup(tag):
    prompts = {
        "python_help": "Please paste the Python code and the full error message, or tell me what you want the code to do.",
        "ai_ml": "Which topic should we cover: AI basics, machine learning, model evaluation, or project ideas?",
        "resume_help": "Share the job role and the resume section you want to improve.",
        "interview_prep": "Tell me the job role and whether you want HR or technical questions.",
        "internships": "Tell me your degree, graduation year, preferred location, and internship domain. Verify openings on official company pages.",
        "career_guidance": "Tell me your interests, strongest skills, and whether you prefer coding or non-coding work.",
        "project_help": "Tell me your project title, dataset, and what you have completed so far.",
        "study_tips": "Tell me the subject, exam date, and how much time you can study each day.",
    }
    return prompts.get(tag, "Please add a little more detail so I can guide you better.")

def recognize_intent(text):
    if not training_patterns or vectorizer is None:
        return "fallback", 0.0
    user_vector = vectorizer.transform([text])
    scores = cosine_similarity(user_vector, pattern_matrix)[0]
    best_index = int(scores.argmax())
    return pattern_tags[best_index], float(scores[best_index])

def chatbot_reply(text):
    tag, confidence = recognize_intent(text)
    # A conservative threshold avoids presenting a weak match as certain.
    if confidence < 0.18 or tag in FALLBACK_TAGS:
        return ("I’m not fully sure I understood that. I can help with study tips, Python, AI/ML, resumes, "
                "interview preparation, internships, career guidance, and student projects.\n\n"
                + helpful_followup("fallback"))
    intent = intent_by_tag.get(tag, {})
    responses = intent.get("responses", [])
    base = random.choice(responses) if responses else "I can help you explore that topic."
    if tag in {"greeting", "goodbye", "thanks"}:
        return base
    return f"{base}\n\n**To give you a more specific answer:** {helpful_followup(tag)}"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I’m your Student Assistant. 🎓\n\nAsk me about studying, Python, AI/ML, resumes, internships, interviews, careers, or student projects."}]

st.title("🎓 Student Assistant Chatbot")
st.write("A free, local intent-based chatbot for student learning and career guidance.")
st.caption("Uses your intents.json dataset and TF-IDF similarity. No Ollama, paid API, or API key required.")

with st.sidebar:
    st.header("Supported topics")
    for item in intents:
        st.markdown(f"- {item.get('tag', 'general').replace('_', ' ').title()}")
    st.divider()
    st.write("**How it works**")
    st.caption("The chatbot compares your message with example phrases in intents.json. Its answer quality depends on the patterns and responses you provide.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! What would you like help with today?"}]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about learning or careers...")
if user_input and user_input.strip():
    user_input = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Finding the closest matching topic..."):
            answer = chatbot_reply(user_input)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.metric("Dataset intents", len(intents))
with col2:
    st.metric("Questions asked", sum(m["role"] == "user" for m in st.session_state.messages))
st.caption("WeIntern Week 2 · Student Assistant Chatbot")
