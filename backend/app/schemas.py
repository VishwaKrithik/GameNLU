from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

EmbeddingModel = Literal[
    "fasttext",
    "word2vec",
    "bert",
    "sentence_transformers",
]

SlotMode = Literal["rule_based", "neural"]


class PredictRequest(BaseModel):
    raw_text: Optional[str] = Field(default=None)
    tokenized_sentences: Optional[List[List[str]]] = Field(default=None)
    embedding_model: EmbeddingModel = "sentence_transformers"
    slot_mode: SlotMode = "rule_based"


class PredictionItem(BaseModel):
    intent: str
    slots: Dict[str, List[str]]


class PredictResponse(BaseModel):
    processed_sentences: List[List[str]]
    embedding_model: str
    slot_mode: str
    results: List[PredictionItem]