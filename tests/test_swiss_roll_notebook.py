"""Execute the maintained notebook, including its CLI subprocesses, on CPU."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import selfsne


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "swiss_roll.ipynb"


@pytest.mark.parametrize("plotting", [False, True])
def test_swiss_roll_notebook_runs_top_to_bottom(tmp_path, plotting):
    if plotting:
        pytest.importorskip("matplotlib")
    # Isolate imports and execute every code cell as ordinary Python. The false
    # branch also verifies the optional plotting path in environments with plots.
    runner = """
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

tempfile.tempdir = sys.argv[3]
if sys.argv[2] == 'no-plots':
    # find_spec is the notebook's optional-feature check; returning None exactly
    # exercises its absent-package branch, without disrupting optional imports
    # inside dependencies which may already know Matplotlib is installed.
    original = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *a, **kw: None if name == 'matplotlib' else original(name, *a, **kw)

notebook = json.loads(Path(sys.argv[1]).read_text())
namespace = {'__name__': '__main__'}
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        assert cell['execution_count'] is None and cell['outputs'] == []
        exec(compile(''.join(cell['source']), f'notebook-cell-{i}', 'exec'), namespace)
assert namespace['SAMPLES'] % namespace['BATCH_SIZE'] == 1
assert namespace['ACCELERATOR'] == 'cpu'
assert namespace['PLOT_AVAILABLE'] == (sys.argv[2] == 'plots')
if namespace['PLOT_AVAILABLE']:
    assert namespace['PLOT_PATH'].stat().st_size > 1000
else:
    assert 'plt' not in namespace and 'PLOT_PATH' not in namespace
print('NOTEBOOK_CHECK_PASSED')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            runner,
            str(NOTEBOOK),
            "plots" if plotting else "no-plots",
            str(tmp_path),
        ],
        cwd=ROOT / "notebooks",
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(Path(selfsne.__file__).resolve().parent.parent),
        },
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NOTEBOOK_CHECK_PASSED" in result.stdout
