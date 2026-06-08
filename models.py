import joblib
import torch
from groq import Groq
from transformers import pipeline as hf_pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

import config

print("Loading language detection model...")
lang_model = joblib.load(config.LANG_MODEL_PATH)

print("Loading emotion classifier from HF Hub...")


emotion_pipe = hf_pipeline(
    "text-classification",
    model=config.EMOTION_MODEL_ID,
    tokenizer=config.EMOTION_MODEL_ID,
    device=0 if torch.cuda.is_available() else -1,
)
print("Loading sentence embedder...")
embedder = SentenceTransformer(config.EMBED_MODEL)

print("Connecting to Qdrant...")
qdrant = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

print("Connecting to Groq...")
groq_client = Groq(api_key=config.GROQ_API_KEY)

print("All models loaded.\n")