import json
from datetime import datetime
from pathlib import Path


def save_results(results, output_dir: Path, mode: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = output_dir / f"{mode}_{timestamp}.json"

    payload = {
        "mode": mode,
        "scores": results.to_pandas().to_dict(),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path