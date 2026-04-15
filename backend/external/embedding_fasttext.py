import numpy as np
from numpy.linalg import norm
import gensim.downloader as api


class IntentEmbedder:
    def __init__(self):
        self.model = api.load("fasttext-wiki-news-subwords-300")

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
        vectors = []
        for token in tokens:
            token = token.lower().strip()
            vectors.append(self.model[token])  # FastText supports subwords
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

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