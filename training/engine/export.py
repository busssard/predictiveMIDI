import json
from pathlib import Path
import numpy as np


def export_model(grid, vocabulary, output_dir):
    """Export a trained grid to binary files for WebGL.

    Writes:
        state.bin -- flat float32 array (H * W * 4)
        weights.bin -- flat float32 array (H * W * 4)
        params.bin -- flat float32 array (H * W * 4)
        config.json -- grid size, vocabulary, etc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = np.array(grid.state, dtype=np.float32)
    weights = np.array(grid.weights, dtype=np.float32)
    params = np.array(grid.params, dtype=np.float32)

    state.tofile(output_dir / "state.bin")
    weights.tofile(output_dir / "weights.bin")
    params.tofile(output_dir / "params.bin")

    config = {
        "grid_size": int(state.shape[0]),
        "num_instruments": len(vocabulary),
        "vocabulary": vocabulary,
        "channels_per_texture": 4,
        "dtype": "float32",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_exported_model(model_dir):
    """Load exported binary model back into numpy arrays (for verification)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    size = config["grid_size"]
    shape = (size, size, 4)

    return {
        "state": np.fromfile(model_dir / "state.bin", dtype=np.float32).reshape(shape),
        "weights": np.fromfile(model_dir / "weights.bin", dtype=np.float32).reshape(shape),
        "params": np.fromfile(model_dir / "params.bin", dtype=np.float32).reshape(shape),
        "config": config,
    }
