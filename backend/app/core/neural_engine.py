from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from app.config import MODEL_PATH


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "<PAD_TAG>"

train_data = [
    {"intent": "attack", "tokens": ["attack", "goblin"], "tags": ["O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["slash", "red", "goblin"], "tags": ["O", "B-TARGET", "I-TARGET"]},
    {"intent": "attack", "tokens": ["hit", "the", "orc"], "tags": ["O", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["attack", "nearest", "enemy"], "tags": ["O", "B-SELECTOR", "B-TARGET"]},
    {"intent": "interact", "tokens": ["open", "chest"], "tags": ["O", "B-OBJECT"]},
    {"intent": "interact", "tokens": ["talk", "to", "merchant"], "tags": ["O", "O", "B-NPC"]},
    {"intent": "interact", "tokens": ["pick", "up", "golden", "key"], "tags": ["O", "O", "B-ITEM", "I-ITEM"]},
    {"intent": "move", "tokens": ["move", "forward"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["go", "left"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["move", "to", "bridge"], "tags": ["O", "O", "B-LOCATION"]},
]

all_words = {PAD_TOKEN, UNK_TOKEN}
all_tags = {PAD_TAG}
all_intents = set()

for sample in train_data:
    all_words.update(sample["tokens"])
    all_tags.update(sample["tags"])
    all_intents.add(sample["intent"])

word2idx = {word: idx for idx, word in enumerate(sorted(all_words))}
idx2word = {idx: word for word, idx in word2idx.items()}

tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

intent2idx = {intent: idx for idx, intent in enumerate(sorted(all_intents))}
idx2intent = {idx: intent for intent, idx in intent2idx.items()}

PAD_WORD_IDX = word2idx[PAD_TOKEN]
UNK_WORD_IDX = word2idx[UNK_TOKEN]
PAD_TAG_IDX = tag2idx[PAD_TAG]


def encode_tokens(tokens: List[str]) -> torch.Tensor:
    return torch.tensor([word2idx.get(tok, UNK_WORD_IDX) for tok in tokens], dtype=torch.long)


class JointNLUModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_tags: int, num_intents: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.slot_classifier = nn.Linear(hidden_dim * 2, num_tags)
        self.intent_classifier = nn.Linear(hidden_dim * 2, num_intents)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        slot_logits = self.slot_classifier(lstm_out)
        sent_repr = torch.mean(lstm_out, dim=1)
        intent_logits = self.intent_classifier(sent_repr)
        return slot_logits, intent_logits


def bio_tags_to_slots(tokens: List[str], tags: List[str]) -> Dict[str, List[str]]:
    slots = defaultdict(list)
    current_slot_name = None
    current_tokens: List[str] = []

    for token, tag in zip(tokens, tags):
        if tag in {"O", PAD_TAG}:
            if current_slot_name and current_tokens:
                slots[current_slot_name].append(" ".join(current_tokens))
            current_slot_name = None
            current_tokens = []
            continue

        prefix, slot_name = tag.split("-", 1)
        slot_name = slot_name.lower()

        if prefix == "B":
            if current_slot_name and current_tokens:
                slots[current_slot_name].append(" ".join(current_tokens))
            current_slot_name = slot_name
            current_tokens = [token]
        elif prefix == "I":
            if current_slot_name == slot_name:
                current_tokens.append(token)
            else:
                if current_slot_name and current_tokens:
                    slots[current_slot_name].append(" ".join(current_tokens))
                current_slot_name = slot_name
                current_tokens = [token]

    if current_slot_name and current_tokens:
        slots[current_slot_name].append(" ".join(current_tokens))

    return dict(slots)


@lru_cache(maxsize=1)
def load_neural_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointNLUModel(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_tags=len(tag2idx),
        num_intents=len(intent2idx),
        pad_idx=PAD_WORD_IDX,
    ).to(device)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model at {MODEL_PATH}. Train once and save state_dict before using neural mode."
        )

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


class NeuralSlotService:
    def __init__(self):
        self.model, self.device = load_neural_model()

    def predict(self, tokens: List[str]):
        x = encode_tokens(tokens).unsqueeze(0).to(self.device)
        with torch.no_grad():
            slot_logits, intent_logits = self.model(x)
            pred_slot_ids = torch.argmax(slot_logits, dim=-1).squeeze(0).tolist()
            pred_intent_id = torch.argmax(intent_logits, dim=-1).item()

        pred_tags = [idx2tag[idx] for idx in pred_slot_ids[: len(tokens)]]
        pred_intent = idx2intent[pred_intent_id]
        pred_slots = bio_tags_to_slots(tokens, pred_tags)
        return {"intent": pred_intent, "slots": pred_slots, "tags": pred_tags}