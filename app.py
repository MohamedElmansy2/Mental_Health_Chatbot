"""
Mental Health Support Chatbot — FastAPI entry point

Run:
    uvicorn app:app --reload

Then open: http://127.0.0.1:8000
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

import config
import models  
from emotion_classifier import detect_emotion
from language_detection import detect_language
from intent_classifier import classify_intent
from rag import rag_answer

app = FastAPI(title="Mental Health Support Chatbot", version="1.0")

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "chat.html"))


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message:  str
    language: str
    emotion:  str
    intent:   str
    answer:   str


_DIRECT_REPLIES_EN = {
    "greeting":    "Hello! I'm here to support you. How are you feeling today?",
    "goodbye":     "Take care of yourself. Remember, support is always here when you need it.",
    "gratitude":   "I'm glad I could help. Please don't hesitate to reach out anytime.",
    "out_of_scope": (
        "I'm a mental health support assistant. "
        "I can only help with questions related to mental health, stress, anxiety, or depression. "
        "Is there something on that topic I can help you with?"
    ),
}


def _localise_reply(english_reply: str, language: str) -> str:
    """Translate a canned English reply into the user's language if needed."""
    if language.lower() in ("english", "en"):
        return english_reply
    resp = models.groq_client.chat.completions.create(
        model=config.INTENT_LLM_MODEL,
        messages=[
            {"role": "system", "content": f"Translate the following text to {language}. Return ONLY the translation."},
            {"role": "user",   "content": english_reply},
        ],
        temperature=0,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()





@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.message.strip()

    language = detect_language(text)
    emotion = detect_emotion(text, language)
    intent   = classify_intent(text)

    if intent == "asking_mental_health_question":
        answer = rag_answer(text, emotion=emotion, language=language)
    else:
        english = _DIRECT_REPLIES_EN.get(intent, _DIRECT_REPLIES_EN["out_of_scope"])
        answer  = _localise_reply(english, language)

    return ChatResponse(
        message=text,
        language=language,
        emotion=emotion,
        intent=intent,
        answer=answer,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
