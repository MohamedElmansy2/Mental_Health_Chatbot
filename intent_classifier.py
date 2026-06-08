import config
import models

_SYSTEM_PROMPT = """You are an intent classifier for a mental health support chatbot.
Classify the user message into exactly one of these intents:
- greeting
- goodbye
- gratitude
- asking_mental_health_question
- out_of_scope

Reply with ONLY the intent label, nothing else.

Examples:
User: Hello there! -> greeting
User: Goodbye, take care -> goodbye
User: Thank you so much -> gratitude
User: I have been feeling very anxious lately -> asking_mental_health_question
User: What is the best recipe for pasta? -> out_of_scope
"""

_VALID_INTENTS = {
    "greeting", "goodbye", "gratitude",
    "asking_mental_health_question", "out_of_scope",
}


def classify_intent(text: str) -> str:
    resp = models.groq_client.chat.completions.create(
        model=config.INTENT_LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": text},
        ],
        temperature=0,
        max_tokens=10,
    )
    label = resp.choices[0].message.content.strip().lower()
    return label if label in _VALID_INTENTS else "out_of_scope"
