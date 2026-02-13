"""Execute the demo notebook code cells sequentially as a smoke check."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def main() -> int:
    notebook_path = Path(__file__).with_name("DonorCompass_Colab_Demo.ipynb")
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        return 1

    with notebook_path.open("r", encoding="utf-8") as fh:
        notebook = json.load(fh)

    namespace: dict = {"__name__": "__main__"}
    code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    print(f"Running {len(code_cells)} code cells from {notebook_path.name} ...")

    for idx, cell in enumerate(code_cells):
        source = _cell_source(cell)
        if not source.strip():
            continue
        try:
            exec(compile(source, f"{notebook_path.name}:cell_{idx}", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - explicit smoke-check error display
            print(f"\nNotebook execution failed in code cell index {idx}")
            print(f"Error: {exc}")
            return 1

    print("Notebook smoke check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
