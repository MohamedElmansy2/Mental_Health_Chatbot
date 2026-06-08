import config
import models

_RAG_SYSTEM = """You are an empathetic mental health support assistant.
You are given retrieved counseling examples to help guide your response.
Give a warm, supportive, and grounded answer.
Do NOT diagnose. Do NOT prescribe medication. Encourage professional help when needed.
Be concise — 3 to 5 sentences.
"""


def _translate_to_english(text: str, language: str) -> str:
    """Translate non-English text to English using Groq. Returns original if already English."""
    if language.lower() in ("english", "en"):
        return text
    resp = models.groq_client.chat.completions.create(
        model=config.INTENT_LLM_MODEL,   # fast model is enough for translation
        messages=[
            {"role": "system", "content": "Translate the following text to English. Return ONLY the translated text, nothing else."},
            {"role": "user",   "content": text},
        ],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def _retrieve(query_in_english: str) -> list[dict]:
    """Embed an English query and return top-k matching Q&A pairs from Qdrant."""
    vec = models.embedder.encode(query_in_english).tolist()
    hits = models.qdrant.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vec,
        limit=config.RAG_TOP_K,
    ).points
    return [{"context": h.payload["context"], "response": h.payload["response"]} for h in hits]


def rag_answer(query: str, emotion: str, language: str) -> str:
    english_query = _translate_to_english(query, language)
    docs = _retrieve(english_query)
    ctx = "\n\n".join(
        f"Example {i+1}:\nUser: {d['context']}\nCounselor: {d['response']}"
        for i, d in enumerate(docs)
    )

    user_msg = (
        f"Detected emotion: {emotion}\n"
        f"User language: {language}\n"
        f"Original user message: {query}\n\n"
        f"Relevant counseling examples (in English):\n{ctx}\n\n"
        f"IMPORTANT: You MUST respond in {language}, not in English."
    )
    resp = models.groq_client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": _RAG_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()
