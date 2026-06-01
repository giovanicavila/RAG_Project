import csv
import json
from pathlib import Path


def load_questions(path: str | Path) -> list[dict]:
    path = Path(path)

    if path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported format: {path.suffix}")
