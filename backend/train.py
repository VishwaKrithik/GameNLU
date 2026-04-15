import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from app.config import MODEL_PATH
from app.core.neural_engine import (
    JointNLUModel,
    PAD_TAG_IDX,
    PAD_WORD_IDX,
    encode_tokens,
    intent2idx,
    tag2idx,
    train_data,
    word2idx,
)


def encode_tags(tags):
    return torch.tensor([tag2idx[tag] for tag in tags], dtype=torch.long)


def encode_intent(intent):
    return torch.tensor(intent2idx[intent], dtype=torch.long)


def main():
    x_list = []
    slot_y_list = []
    intent_y_list = []

    for sample in train_data:
        x_list.append(encode_tokens(sample["tokens"]))
        slot_y_list.append(encode_tags(sample["tags"]))
        intent_y_list.append(encode_intent(sample["intent"]))

    x_padded = pad_sequence(x_list, batch_first=True, padding_value=PAD_WORD_IDX)
    slot_y_padded = pad_sequence(slot_y_list, batch_first=True, padding_value=PAD_TAG_IDX)
    intent_y = torch.stack(intent_y_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JointNLUModel(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_tags=len(tag2idx),
        num_intents=len(intent2idx),
        pad_idx=PAD_WORD_IDX,
    ).to(device)

    x_padded = x_padded.to(device)
    slot_y_padded = slot_y_padded.to(device)
    intent_y = intent_y.to(device)

    slot_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG_IDX)
    intent_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        slot_logits, intent_logits = model(x_padded)
        slot_loss = slot_criterion(slot_logits.view(-1, len(tag2idx)), slot_y_padded.view(-1))
        intent_loss = intent_criterion(intent_logits, intent_y)
        loss = slot_loss + intent_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()