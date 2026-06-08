import config
import models


def detect_emotion(text: str, language: str = "en") -> str:
    if language.lower() not in ("en", "english"):
        translated = models.groq_client.chat.completions.create(
            model=config.INTENT_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Translate this text to English. Return ONLY the translation."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0,
            max_tokens=100
        )

        text = translated.choices[0].message.content.strip()

    result = models.emotion_pipe(
        text,
        truncation=True,
        max_length=128
    )[0]

    return result["label"].lower()