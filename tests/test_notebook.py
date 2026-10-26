"""The control-tower notebook must run top to bottom (CPU) and only go through
mslm.tower, so what it shows is what the shell scripts do."""

import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "control_tower.ipynb"


def test_notebook_uses_tower_only():
    nb = json.loads(NB.read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "sbatch_command" in src and "tower.show" in src and "source_of" in src
    for forbidden in ["subprocess", "os.system", "sbatch --", "!sbatch"]:
        assert forbidden not in src, f"notebook must not submit or hand-build commands ({forbidden})"


def _cells(text):
    return [(c["cell_type"], "".join(c["source"])) for c in json.loads(text)["cells"]]


def test_notebook_up_to_date():
    import subprocess
    import sys
    before = NB.read_text()
    try:
        subprocess.run([sys.executable, str(REPO / "tools/build_notebook.py")], check=True, capture_output=True)
        assert _cells(NB.read_text()) == _cells(before), "control_tower.ipynb differs from tools/build_notebook.py"
    finally:
        NB.write_text(before)


def test_notebook_executes(monkeypatch):
    nbclient = pytest.importorskip("nbclient")
    import nbformat
    nb = nbformat.read(NB, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(REPO)}})
    client.execute()
    outs = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "sbatch" in outs and "slurm/train.sbatch" in outs
