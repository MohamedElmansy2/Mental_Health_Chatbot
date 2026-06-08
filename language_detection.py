import re
import models


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def detect_language(text: str) -> str:
    return models.lang_model.predict([_clean(text)])[0]
