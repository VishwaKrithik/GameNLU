import re
from typing import List


STOP_CONNECTORS = {"and", "then", "but", "also", "next"}


def simple_preprocess(raw_text: str) -> List[List[str]]:
    text = raw_text.lower().strip()
    if not text:
        return []

    segments = re.split(r"\b(?:then|but|also|next|after that)\b", text)
    processed = []

    for segment in segments:
        tokens = re.findall(r"[a-zA-Z0-9_]+", segment.lower())
        tokens = [t for t in tokens if t not in STOP_CONNECTORS]
        if tokens:
            processed.append(tokens)

    return processed