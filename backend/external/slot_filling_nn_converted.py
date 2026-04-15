from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

INPUT_FILE = "data_preprocessing.json"
OUTPUT_FILE = "output.json"
MODEL_FILE = "slot_filling_model.pt"

train_data = [
    # -------------------------
    # ATTACK INTENT
    # -------------------------
    {"intent": "attack", "tokens": ["attack", "goblin"], "tags": ["O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["slash", "red", "goblin"], "tags": ["O", "B-TARGET", "I-TARGET"]},
    {"intent": "attack", "tokens": ["hit", "the", "orc"], "tags": ["O", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["strike", "ancient", "stone", "golem"], "tags": ["O", "B-TARGET", "I-TARGET", "I-TARGET"]},
    {"intent": "attack", "tokens": ["shoot", "blue", "forest", "bat"], "tags": ["O", "B-TARGET", "I-TARGET", "I-TARGET"]},
    {"intent": "attack", "tokens": ["stab", "poison", "spider", "twice"], "tags": ["O", "B-TARGET", "I-TARGET", "B-COUNT"]},
    {"intent": "attack", "tokens": ["attack", "enemy", "3", "times"], "tags": ["O", "B-TARGET", "B-COUNT", "O"]},
    {"intent": "attack", "tokens": ["attack", "enemy", "three", "times"], "tags": ["O", "B-TARGET", "B-COUNT", "O"]},
    {"intent": "attack", "tokens": ["hit", "goblin", "once"], "tags": ["O", "B-TARGET", "B-COUNT"]},
    {"intent": "attack", "tokens": ["slash", "orc", "1", "time"], "tags": ["O", "B-TARGET", "B-COUNT", "O"]},
    {"intent": "attack", "tokens": ["shoot", "bat", "five", "times"], "tags": ["O", "B-TARGET", "B-COUNT", "O"]},
    {"intent": "attack", "tokens": ["fire", "arrow", "at", "goblin"], "tags": ["O", "B-WEAPON", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["throw", "spear", "at", "red", "orc"], "tags": ["O", "B-WEAPON", "O", "B-TARGET", "I-TARGET"]},
    {"intent": "attack", "tokens": ["use", "dagger", "on", "enemy"], "tags": ["O", "B-WEAPON", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["swing", "iron", "sword", "at", "skeleton"], "tags": ["O", "B-WEAPON", "I-WEAPON", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["cast", "fireball", "on", "goblin"], "tags": ["O", "B-SKILL", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["cast", "ice", "blast", "on", "blue", "slime"], "tags": ["O", "B-SKILL", "I-SKILL", "O", "B-TARGET", "I-TARGET"]},
    {"intent": "attack", "tokens": ["use", "poison", "cloud", "against", "orc"], "tags": ["O", "B-SKILL", "I-SKILL", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["attack", "nearest", "enemy"], "tags": ["O", "B-SELECTOR", "B-TARGET"]},
    {"intent": "attack", "tokens": ["attack", "closest", "red", "goblin"], "tags": ["O", "B-SELECTOR", "B-TARGET", "I-TARGET"]},
    {"intent": "attack", "tokens": ["attack", "leftmost", "archer"], "tags": ["O", "B-SELECTOR", "B-TARGET"]},
    {"intent": "attack", "tokens": ["hit", "the", "weakest", "enemy"], "tags": ["O", "O", "B-SELECTOR", "B-TARGET"]},
    {"intent": "attack", "tokens": ["shoot", "the", "enemy", "behind", "the", "barrel"], "tags": ["O", "O", "B-TARGET", "O", "O", "B-REFERENCE"]},
    {"intent": "attack", "tokens": ["attack", "goblin", "near", "bridge"], "tags": ["O", "B-TARGET", "O", "B-LOCATION"]},
    {"intent": "attack", "tokens": ["shoot", "orc", "at", "north", "gate"], "tags": ["O", "B-TARGET", "O", "B-LOCATION", "I-LOCATION"]},
    {"intent": "attack", "tokens": ["cast", "lightning", "on", "enemy", "by", "the", "tower"], "tags": ["O", "B-SKILL", "O", "B-TARGET", "O", "O", "B-LOCATION"]},
    {"intent": "attack", "tokens": ["attack", "all", "goblins"], "tags": ["O", "B-SCOPE", "B-TARGET"]},
    {"intent": "attack", "tokens": ["hit", "every", "enemy"], "tags": ["O", "B-SCOPE", "B-TARGET"]},
    {"intent": "attack", "tokens": ["shoot", "both", "archers"], "tags": ["O", "B-SCOPE", "B-TARGET"]},
    {"intent": "attack", "tokens": ["attack", "the", "second", "goblin"], "tags": ["O", "O", "B-ORDER", "B-TARGET"]},
    {"intent": "attack", "tokens": ["hit", "the", "last", "orc"], "tags": ["O", "O", "B-ORDER", "B-TARGET"]},
    {"intent": "attack", "tokens": ["strike", "enemy", "with", "hammer"], "tags": ["O", "B-TARGET", "O", "B-WEAPON"]},
    {"intent": "attack", "tokens": ["attack", "the", "big", "red", "dragon", "twice"], "tags": ["O", "O", "B-TARGET", "I-TARGET", "I-TARGET", "B-COUNT"]},
    {"intent": "attack", "tokens": ["use", "meteor", "strike", "on", "the", "boss"], "tags": ["O", "B-SKILL", "I-SKILL", "O", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["attack", "boss", "now"], "tags": ["O", "B-TARGET", "O"]},
    {"intent": "attack", "tokens": ["quickly", "attack", "goblin"], "tags": ["O", "O", "B-TARGET"]},
    {"intent": "attack", "tokens": ["do", "not", "attack", "villager"], "tags": ["O", "O", "O", "B-TARGET"]},

    # -------------------------
    # INTERACT INTENT
    # -------------------------
    {"intent": "interact", "tokens": ["open", "chest"], "tags": ["O", "B-OBJECT"]},
    {"intent": "interact", "tokens": ["open", "wooden", "chest"], "tags": ["O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["unlock", "iron", "door"], "tags": ["O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["close", "north", "gate"], "tags": ["O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["pull", "lever"], "tags": ["O", "B-OBJECT"]},
    {"intent": "interact", "tokens": ["push", "stone", "block"], "tags": ["O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["break", "barrel"], "tags": ["O", "B-OBJECT"]},
    {"intent": "interact", "tokens": ["light", "torch"], "tags": ["O", "B-OBJECT"]},
    {"intent": "interact", "tokens": ["extinguish", "wall", "torch"], "tags": ["O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["read", "ancient", "scroll"], "tags": ["O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["inspect", "strange", "statue"], "tags": ["O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["touch", "blue", "crystal"], "tags": ["O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["talk", "to", "merchant"], "tags": ["O", "O", "B-NPC"]},
    {"intent": "interact", "tokens": ["speak", "to", "old", "wizard"], "tags": ["O", "O", "B-NPC", "I-NPC"]},
    {"intent": "interact", "tokens": ["trade", "with", "village", "merchant"], "tags": ["O", "O", "B-NPC", "I-NPC"]},
    {"intent": "interact", "tokens": ["give", "potion", "to", "healer"], "tags": ["O", "B-ITEM", "O", "B-NPC"]},
    {"intent": "interact", "tokens": ["show", "key", "to", "guard"], "tags": ["O", "B-ITEM", "O", "B-NPC"]},
    {"intent": "interact", "tokens": ["pick", "up", "golden", "key"], "tags": ["O", "O", "B-ITEM", "I-ITEM"]},
    {"intent": "interact", "tokens": ["take", "silver", "coin"], "tags": ["O", "B-ITEM", "I-ITEM"]},
    {"intent": "interact", "tokens": ["drop", "old", "map"], "tags": ["O", "B-ITEM", "I-ITEM"]},
    {"intent": "interact", "tokens": ["use", "rusty", "key", "on", "cell", "door"], "tags": ["O", "B-ITEM", "I-ITEM", "O", "B-OBJECT", "I-OBJECT"]},
    {"intent": "interact", "tokens": ["use", "health", "potion"], "tags": ["O", "B-ITEM", "I-ITEM"]},
    {"intent": "interact", "tokens": ["drink", "mana", "potion"], "tags": ["O", "B-ITEM", "I-ITEM"]},
    {"intent": "interact", "tokens": ["equip", "iron", "sword"], "tags": ["O", "B-ITEM", "I-ITEM"]},
    {"intent": "interact", "tokens": ["wear", "steel", "helmet"], "tags": ["O", "B-ITEM", "I-ITEM"]},
    {"intent": "interact", "tokens": ["remove", "magic", "ring"], "tags": ["O", "B-ITEM", "I-ITEM"]},
    {"intent": "interact", "tokens": ["inspect", "inventory"], "tags": ["O", "B-MENU"]},
    {"intent": "interact", "tokens": ["open", "map"], "tags": ["O", "B-MENU"]},
    {"intent": "interact", "tokens": ["check", "quest", "log"], "tags": ["O", "B-MENU", "I-MENU"]},
    {"intent": "interact", "tokens": ["talk", "to", "guard", "near", "gate"], "tags": ["O", "O", "B-NPC", "O", "B-LOCATION"]},
    {"intent": "interact", "tokens": ["open", "chest", "inside", "tower"], "tags": ["O", "B-OBJECT", "O", "B-LOCATION"]},
    {"intent": "interact", "tokens": ["pick", "up", "key", "from", "table"], "tags": ["O", "O", "B-ITEM", "O", "B-LOCATION"]},
    {"intent": "interact", "tokens": ["buy", "torch", "from", "merchant"], "tags": ["O", "B-ITEM", "O", "B-NPC"]},
    {"intent": "interact", "tokens": ["sell", "gem", "to", "trader"], "tags": ["O", "B-ITEM", "O", "B-NPC"]},
    {"intent": "interact", "tokens": ["use", "lever", "twice"], "tags": ["O", "B-OBJECT", "B-COUNT"]},
    {"intent": "interact", "tokens": ["open", "three", "chests"], "tags": ["O", "B-COUNT", "B-OBJECT"]},
    {"intent": "interact", "tokens": ["inspect", "all", "barrels"], "tags": ["O", "B-SCOPE", "B-OBJECT"]},
    {"intent": "interact", "tokens": ["use", "the", "second", "key"], "tags": ["O", "O", "B-ORDER", "B-ITEM"]},
    {"intent": "interact", "tokens": ["open", "the", "last", "door"], "tags": ["O", "O", "B-ORDER", "B-OBJECT"]},

    # -------------------------
    # MOVE INTENT
    # -------------------------
    {"intent": "move", "tokens": ["move", "forward"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["go", "left"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["walk", "right"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["run", "backward"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["move", "north"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["move", "south"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["go", "east"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["head", "west"], "tags": ["O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["move", "to", "bridge"], "tags": ["O", "O", "B-LOCATION"]},
    {"intent": "move", "tokens": ["go", "to", "castle", "gate"], "tags": ["O", "O", "B-LOCATION", "I-LOCATION"]},
    {"intent": "move", "tokens": ["walk", "toward", "the", "tower"], "tags": ["O", "O", "O", "B-LOCATION"]},
    {"intent": "move", "tokens": ["run", "to", "old", "camp"], "tags": ["O", "O", "B-LOCATION", "I-LOCATION"]},
    {"intent": "move", "tokens": ["move", "forward", "3", "steps"], "tags": ["O", "B-DIRECTION", "B-DISTANCE", "B-UNIT"]},
    {"intent": "move", "tokens": ["go", "left", "two", "tiles"], "tags": ["O", "B-DIRECTION", "B-DISTANCE", "B-UNIT"]},
    {"intent": "move", "tokens": ["walk", "north", "5", "meters"], "tags": ["O", "B-DIRECTION", "B-DISTANCE", "B-UNIT"]},
    {"intent": "move", "tokens": ["run", "east", "one", "square"], "tags": ["O", "B-DIRECTION", "B-DISTANCE", "B-UNIT"]},
    {"intent": "move", "tokens": ["move", "up", "the", "stairs"], "tags": ["O", "B-DIRECTION", "O", "B-LOCATION"]},
    {"intent": "move", "tokens": ["move", "down", "the", "stairs"], "tags": ["O", "B-DIRECTION", "O", "B-LOCATION"]},
    {"intent": "move", "tokens": ["go", "up", "2", "floors"], "tags": ["O", "B-DIRECTION", "B-DISTANCE", "B-UNIT"]},
    {"intent": "move", "tokens": ["walk", "down", "one", "level"], "tags": ["O", "B-DIRECTION", "B-DISTANCE", "B-UNIT"]},
    {"intent": "move", "tokens": ["quickly", "move", "forward", "4"], "tags": ["O", "O", "B-DIRECTION", "B-DISTANCE"]},
    {"intent": "move", "tokens": ["carefully", "step", "left"], "tags": ["O", "O", "B-DIRECTION"]},
    {"intent": "move", "tokens": ["do", "not", "move"], "tags": ["O", "O", "O"]},
    {"intent": "move", "tokens": ["stop"], "tags": ["O"]},
    {"intent": "move", "tokens": ["wait", "here"], "tags": ["O", "B-LOCATION"]},
    {"intent": "move", "tokens": ["return", "to", "spawn", "point"], "tags": ["O", "O", "B-LOCATION", "I-LOCATION"]},
    {"intent": "move", "tokens": ["follow", "the", "merchant"], "tags": ["O", "O", "B-NPC"]},
    {"intent": "move", "tokens": ["move", "toward", "the", "guard"], "tags": ["O", "O", "O", "B-NPC"]},
]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "<PAD_TAG>"

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


def encode_tags(tags: List[str]) -> torch.Tensor:
    return torch.tensor([tag2idx[tag] for tag in tags], dtype=torch.long)


def encode_intent(intent: str) -> torch.Tensor:
    return torch.tensor(intent2idx[intent], dtype=torch.long)


X_list = []
slot_y_list = []
intent_y_list = []

for sample in train_data:
    tokens = sample["tokens"]
    tags = sample["tags"]
    intent = sample["intent"]

    assert len(tokens) == len(tags), f"Token/tag length mismatch in sample: {sample}"

    X_list.append(encode_tokens(tokens))
    slot_y_list.append(encode_tags(tags))
    intent_y_list.append(encode_intent(intent))

X_padded = pad_sequence(X_list, batch_first=True, padding_value=PAD_WORD_IDX)
slot_y_padded = pad_sequence(slot_y_list, batch_first=True, padding_value=PAD_TAG_IDX)
intent_y = torch.stack(intent_y_list)


class JointNLUModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_tags: int,
        num_intents: int,
        pad_idx: int,
    ):
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
        if tag == "O" or tag == PAD_TAG:
            if current_slot_name is not None and current_tokens:
                slots[current_slot_name].append(" ".join(current_tokens))
                current_slot_name = None
                current_tokens = []
            continue

        if "-" not in tag:
            if current_slot_name is not None and current_tokens:
                slots[current_slot_name].append(" ".join(current_tokens))
            current_slot_name = None
            current_tokens = []
            continue

        prefix, slot_name = tag.split("-", 1)
        slot_name = slot_name.lower()

        if prefix == "B":
            if current_slot_name is not None and current_tokens:
                slots[current_slot_name].append(" ".join(current_tokens))
            current_slot_name = slot_name
            current_tokens = [token]
        elif prefix == "I":
            if current_slot_name == slot_name:
                current_tokens.append(token)
            else:
                if current_slot_name is not None and current_tokens:
                    slots[current_slot_name].append(" ".join(current_tokens))
                current_slot_name = slot_name
                current_tokens = [token]

    if current_slot_name is not None and current_tokens:
        slots[current_slot_name].append(" ".join(current_tokens))

    return dict(slots)


def predict(model: JointNLUModel, device: torch.device, sentence_tokens: List[str]) -> Dict[str, object]:
    model.eval()
    x = encode_tokens(sentence_tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        slot_logits, intent_logits = model(x)
        pred_slot_ids = torch.argmax(slot_logits, dim=-1).squeeze(0).tolist()
        pred_intent_id = torch.argmax(intent_logits, dim=-1).item()

    pred_tags = [idx2tag[idx] for idx in pred_slot_ids[: len(sentence_tokens)]]
    pred_intent = idx2intent[pred_intent_id]
    pred_slots = bio_tags_to_slots(sentence_tokens, pred_tags)
    return {
        "tokens": sentence_tokens,
        "intent": pred_intent,
        "tags": pred_tags,
        "slots": pred_slots,
    }


def load_input_json(input_path: Path) -> List[List[str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of token lists.")

    cleaned: List[List[str]] = []
    for item in data:
        if not isinstance(item, list):
            raise ValueError("Each entry in input JSON must be a list of tokens.")
        tokens = [str(tok).strip().lower() for tok in item if str(tok).strip()]
        if tokens:
            cleaned.append(tokens)
    return cleaned


def train_model(num_epochs: int = 100) -> Tuple[JointNLUModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JointNLUModel(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_tags=len(tag2idx),
        num_intents=len(intent2idx),
        pad_idx=PAD_WORD_IDX,
    ).to(device)

    x_padded = X_padded.to(device)
    slots_padded = slot_y_padded.to(device)
    intents = intent_y.to(device)

    slot_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG_IDX)
    intent_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        slot_logits, intent_logits = model(x_padded)
        slot_loss = slot_criterion(slot_logits.view(-1, len(tag2idx)), slots_padded.view(-1))
        intent_loss = intent_criterion(intent_logits, intents)
        loss = slot_loss + intent_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                predicted_intents = torch.argmax(intent_logits, dim=1)
                intent_acc = (predicted_intents == intents).float().mean().item()
            print(
                f"Epoch {epoch + 1:>3} | "
                f"Total Loss: {loss.item():.4f} | "
                f"Slot Loss: {slot_loss.item():.4f} | "
                f"Intent Loss: {intent_loss.item():.4f} | "
                f"Intent Acc: {intent_acc:.4f}"
            )

    return model, device


def save_results(results: List[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    input_sentences = load_input_json(input_path)
    print(f"Loaded {len(input_sentences)} sentence(s) from {input_path}")

    model, device = train_model(num_epochs=100)

    results = []
    for sentence_tokens in input_sentences:
        result = predict(model, device, sentence_tokens)
        result_processed = {'intent':result['intent'], 'slots': result['slots']}
        results.append(result_processed)

    save_results(results, output_path)
    print(f"Saved predictions to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
