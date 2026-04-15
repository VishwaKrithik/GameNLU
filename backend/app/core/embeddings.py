from functools import lru_cache
import sys
from pathlib import Path

from app.config import EXTERNAL_DIR

if str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))


@lru_cache(maxsize=8)
def get_embedder(model_name: str):
    if model_name == "fasttext":
        from embedding_fasttext import IntentEmbedder
    elif model_name == "word2vec":
        from embedding_word2vec import IntentEmbedder
    elif model_name == "bert":
        from embedding_bert import IntentEmbedder
    elif model_name == "sentence_transformers":
        from embedding_sentence_transformers import IntentEmbedder
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")

    return IntentEmbedder()