import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ── Model identifiers ─────────────────────────────────────────────────────────
EMOTION_MODEL_ID = os.getenv("EMOTION_MODEL_ID", "your_hf_username/emotion-roberta")
LANG_MODEL_PATH  = os.getenv("LANG_MODEL_PATH",  "models/language_detection_tfidf_linearsvc.pkl")
EMBED_MODEL      = "all-MiniLM-L6-v2"
LLM_MODEL        = "openai/gpt-oss-120b"
INTENT_LLM_MODEL = "llama-3.3-70b-versatile"

# ── Qdrant ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = "mental_health_qa"
RAG_TOP_K       = 3

# ── Validation: warn if required keys are missing ────────────────────────────
_REQUIRED = {
    "GROQ_API_KEY":   GROQ_API_KEY,
    "QDRANT_URL":     QDRANT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
}
for _name, _val in _REQUIRED.items():
    if not _val:
        raise EnvironmentError(
            f"Missing required environment variable: {_name}\n"
            f"Copy .env.example → .env and fill in your values."
        )
