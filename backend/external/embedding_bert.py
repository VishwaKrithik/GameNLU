import numpy as np
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
import torch


class IntentEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.model.eval()

        self.intent_examples = {
            "move": [
                ["move", "forward"],
                ["go", "left"],
                ["walk", "north"],
                ["run", "backward"]
            ],
            "attack": [
                ["attack", "enemy"],
                ["hit", "target"],
                ["strike", "closest", "enemy"],
                ["fight", "monster"]
            ],
            "interact": [
                ["interact", "with", "npc"],
                ["talk", "to", "wizard"],
                ["open", "door"],
                ["use", "lever"]
            ]
        }

        self.intent_embeddings = {
            intent: self._average_example_embeddings(examples)
            for intent, examples in self.intent_examples.items()
        }

    def sentence_embedding(self, tokens):
        text = " ".join(tokens)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**encoded)

        # CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return cls_embedding

    def _average_example_embeddings(self, examples):
        vecs = [self.sentence_embedding(example) for example in examples]
        return np.mean(vecs, axis=0)

    def cosine_similarity(self, v1, v2):
        if norm(v1) == 0 or norm(v2) == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm(v1) * norm(v2)))

    def predict_intent(self, sentence_tokens):
        sent_emb = self.sentence_embedding(sentence_tokens)

        best_intent = None
        best_score = -1.0

        for intent, intent_emb in self.intent_embeddings.items():
            score = self.cosine_similarity(sent_emb, intent_emb)
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent