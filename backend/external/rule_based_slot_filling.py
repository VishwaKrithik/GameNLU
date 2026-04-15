from typing import Dict, List


class SlotFillingEngine:
    def __init__(self):
        self.direction_words = {
            "left", "right", "forward", "forwards",
            "backward", "backwards", "up", "down",
            "north", "south", "east", "west",
            "leftward", "rightward", "upward", "upwards",
            "downward", "downwards"
        }

        self.direction_map = {
            "forwards": "forward",
            "backwards": "backward",
            "leftward": "left",
            "rightward": "right",
            "upwards": "upward",
            "up": "upward",
            "downwards": "downward",
            "down": "downward"
        }

        self.selector_words = {
            "closest", "nearest", "farthest", "furthest",
            "weakest", "strongest", "first", "last"
        }

    def _normalize_direction(self, word: str) -> str:
        return self.direction_map.get(word, word)

    def extract_slots(self, tokens: List[str], intent: str) -> Dict[str, List[str]]:
        tokens = [t.lower().strip() for t in tokens]
        slots: Dict[str, List[str]] = {}

        if intent == "move":
            self._extract_move_slots(tokens, slots)
        elif intent == "attack":
            self._extract_attack_slots(tokens, slots)
        elif intent == "interact":
            self._extract_interact_slots(tokens, slots)

        return slots

    def _extract_move_slots(self, tokens: List[str], slots: Dict[str, List[str]]) -> None:
        for token in tokens:
            if token in self.direction_words:
                slots["direction"] = [self._normalize_direction(token)]
                return

    def _extract_attack_slots(self, tokens: List[str], slots: Dict[str, List[str]]) -> None:
        filtered = [t for t in tokens if t not in {"attack", "the", "a", "an"}]

        for token in filtered:
            if token in self.selector_words:
                slots["selector"] = [token]
                break

        target = None

        for i, token in enumerate(filtered):
            if token in self.selector_words and i + 1 < len(filtered):
                target = filtered[i + 1]
                break

        if target is None:
            for token in filtered:
                if token not in self.selector_words:
                    target = token
                    break

        if target:
            slots["target"] = [target]

    def _extract_interact_slots(self, tokens: List[str], slots: Dict[str, List[str]]) -> None:
        filtered = [t for t in tokens if t not in {"interact", "with", "the", "a", "an"}]
        if filtered:
            slots["target"] = [filtered[0]]