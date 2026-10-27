#!/usr/bin/env python3
"""Copy the bundled difficulty-3 model into the legacy default location."""
from pathlib import Path
import shutil


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "pretrained" / "quantum_skill3_candidate.pt"
    target = root / "outputs" / "models" / "quantum.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an existing trained checkpoint.
    try:
        with source.open("rb") as inp, target.open("xb") as out:
            shutil.copyfileobj(inp, out)
    except FileExistsError:
        print(f"Keeping existing {target.relative_to(root)}")
    else:
        print(f"Initialized {target.relative_to(root)} from {source.relative_to(root)}")


if __name__ == "__main__":
    main()
