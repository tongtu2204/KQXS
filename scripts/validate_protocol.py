"""Validate the canonical split and tracked raw KQXS data."""

import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import FINAL_TEST, TRAIN_DEVELOPMENT, VALIDATION, validate_protocol
from src.data import load_data, validate_data


def main() -> None:
    validate_protocol()
    data = load_data(rebuild=True)
    summary = validate_data(data)
    summary["protocol"] = {
        TRAIN_DEVELOPMENT.name: [
            TRAIN_DEVELOPMENT.start.date().isoformat(),
            TRAIN_DEVELOPMENT.end.date().isoformat(),
        ],
        VALIDATION.name: [
            VALIDATION.start.date().isoformat(),
            VALIDATION.end.date().isoformat(),
        ],
        FINAL_TEST.name: [
            FINAL_TEST.start.date().isoformat(),
            FINAL_TEST.end.date().isoformat(),
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
