"""Tiny deterministic acceptance workload, not a scientific benchmark."""

import json
from pathlib import Path
import sys


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(
        json.dumps({"metrics": {"loss": (config["x"] - 2) ** 2}}) + "\n"
    )


if __name__ == "__main__":
    main()
