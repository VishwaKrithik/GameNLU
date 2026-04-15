import sys
from pathlib import Path

from app.config import EXTERNAL_DIR

if str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))

from rule_based_slot_filling import SlotFillingEngine


class RuleBasedSlotService:
    def __init__(self):
        self.engine = SlotFillingEngine()

    def extract(self, tokens, intent):
        return self.engine.extract_slots(tokens, intent)