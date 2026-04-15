from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = BASE_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "slot_filling_model.pt"
C_PREPROCESSOR_SOURCE = EXTERNAL_DIR / "data_preprocessing.c"
C_PREPROCESSOR_BINARY = EXTERNAL_DIR / "data_preprocessing"