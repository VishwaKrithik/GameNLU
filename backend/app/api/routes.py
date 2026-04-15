from fastapi import APIRouter, HTTPException

from app.core.embeddings import get_embedder
from app.core.neural_engine import NeuralSlotService
from app.core.preprocessing import simple_preprocess
from app.core.rule_based_engine import RuleBasedSlotService
from app.schemas import PredictRequest, PredictResponse
from app.utils.c_preprocessor import CPreprocessorError, preprocess_with_c

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.tokenized_sentences:
        processed_sentences = req.tokenized_sentences
    elif req.raw_text:
        try:
            processed_sentences = preprocess_with_c(req.raw_text)
        except CPreprocessorError:
            processed_sentences = simple_preprocess(req.raw_text)
    else:
        raise HTTPException(status_code=400, detail="Provide either raw_text or tokenized_sentences")

    if not processed_sentences:
        raise HTTPException(status_code=400, detail="No valid tokens found")

    results = []

    if req.slot_mode == "rule_based":
        embedder = get_embedder(req.embedding_model)
        slot_service = RuleBasedSlotService()

        for tokens in processed_sentences:
            intent = embedder.predict_intent(tokens)
            slots = slot_service.extract(tokens, intent)
            results.append({"intent": intent, "slots": slots})

    elif req.slot_mode == "neural":
        slot_service = NeuralSlotService()
        for tokens in processed_sentences:
            neural_result = slot_service.predict(tokens)
            results.append({
                "intent": neural_result["intent"],
                "slots": neural_result["slots"],
            })
    else:
        raise HTTPException(status_code=400, detail="Invalid slot mode")

    return {
        "processed_sentences": processed_sentences,
        "embedding_model": req.embedding_model,
        "slot_mode": req.slot_mode,
        "results": results,
    }