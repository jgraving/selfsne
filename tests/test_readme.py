"""Keep the documented Python workflow executable as the API evolves."""

from pathlib import Path
import re

import numpy as np
import torch


def test_readme_training_prediction_and_reload(tmp_path, monkeypatch):
    readme = Path(__file__).resolve().parents[1] / "README.md"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), flags=re.DOTALL)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for block in blocks:
        exec(compile(block, str(readme), "exec"), namespace)
    actual = namespace["embeddings"]
    assert actual.shape == (128, 2)
    assert np.isfinite(actual).all()
    with torch.inference_mode():
        expected = namespace["model"](namespace["observations"])["target_embedding"]
        restored = namespace["restored"](namespace["observations"])["target_embedding"]
    torch.testing.assert_close(restored, expected)
