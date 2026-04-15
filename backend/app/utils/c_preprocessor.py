import subprocess
from pathlib import Path
from typing import List

from app.config import C_PREPROCESSOR_BINARY, C_PREPROCESSOR_SOURCE, EXTERNAL_DIR


class CPreprocessorError(Exception):
    pass



def compile_c_preprocessor():
    if C_PREPROCESSOR_BINARY.exists():
        return

    compile_cmd = [
        "gcc",
        str(C_PREPROCESSOR_SOURCE),
        "-o",
        str(C_PREPROCESSOR_BINARY),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise CPreprocessorError(result.stderr or "Failed to compile C preprocessor")



def preprocess_with_c(raw_text: str) -> List[List[str]]:
    compile_c_preprocessor()

    result = subprocess.run(
        [str(C_PREPROCESSOR_BINARY)],
        input=raw_text + "\n",
        text=True,
        capture_output=True,
        cwd=EXTERNAL_DIR,
    )

    if result.returncode != 0:
        raise CPreprocessorError(result.stderr or result.stdout or "C preprocessing failed")

    output_file = EXTERNAL_DIR / "data_preprocessing.json"
    if not output_file.exists():
        raise CPreprocessorError("C preprocessor did not generate data_preprocessing.json")

    import json
    with open(output_file, "r", encoding="utf-8") as f:
        return json.load(f)