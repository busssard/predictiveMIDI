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
        fc_weights.bin -- flat float32 (W * H * H) if present
        fc_skip_weights.bin -- flat float32 (W * H * H) if present
        w_temporal.bin -- flat float32 (H * W) if present
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = np.array(grid.state, dtype=np.float32)
    weights = np.array(grid.weights, dtype=np.float32)
    params = np.array(grid.params, dtype=np.float32)

    state.tofile(output_dir / "state.bin")
    weights.tofile(output_dir / "weights.bin")
    params.tofile(output_dir / "params.bin")

    # Export log_precision if present
    if hasattr(grid, 'log_precision') and grid.log_precision is not None:
        log_prec = np.array(grid.log_precision, dtype=np.float32)
        log_prec.tofile(output_dir / "log_precision.bin")

    # Export FC weights if present
    if hasattr(grid, 'fc_weights') and grid.fc_weights is not None:
        fc_w = np.array(grid.fc_weights, dtype=np.float32)
        fc_w.tofile(output_dir / "fc_weights.bin")

    if hasattr(grid, 'fc_skip_weights') and grid.fc_skip_weights is not None:
        fc_sw = np.array(grid.fc_skip_weights, dtype=np.float32)
        fc_sw.tofile(output_dir / "fc_skip_weights.bin")

    # Export w_temporal if present
    if hasattr(grid, 'w_temporal') and grid.w_temporal is not None:
        w_t = np.array(grid.w_temporal, dtype=np.float32)
        w_t.tofile(output_dir / "w_temporal.bin")

    height, width = int(state.shape[0]), int(state.shape[1])
    connectivity = getattr(grid, 'connectivity', 'neighbor')
    config = {
        "grid_width": width,
        "grid_height": height,
        "grid_size": height,  # backward compat (= height)
        "num_instruments": len(vocabulary),
        "vocabulary": vocabulary,
        "channels_per_texture": 4,
        "dtype": "float32",
        "connectivity": connectivity,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))


def load_exported_model(model_dir):
    """Load exported binary model back into numpy arrays (for verification)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    width = config.get("grid_width", config["grid_size"])
    height = config.get("grid_height", config["grid_size"])
    shape = (height, width, 4)

    result = {
        "state": np.fromfile(model_dir / "state.bin", dtype=np.float32).reshape(shape),
        "weights": np.fromfile(model_dir / "weights.bin", dtype=np.float32).reshape(shape),
        "params": np.fromfile(model_dir / "params.bin", dtype=np.float32).reshape(shape),
        "config": config,
    }
    lp_path = model_dir / "log_precision.bin"
    if lp_path.exists():
        result["log_precision"] = np.fromfile(
            lp_path, dtype=np.float32).reshape((height, width))

    fc_path = model_dir / "fc_weights.bin"
    if fc_path.exists():
        result["fc_weights"] = np.fromfile(
            fc_path, dtype=np.float32).reshape((width, height, height))

    fc_skip_path = model_dir / "fc_skip_weights.bin"
    if fc_skip_path.exists():
        result["fc_skip_weights"] = np.fromfile(
            fc_skip_path, dtype=np.float32).reshape((width, height, height))

    wt_path = model_dir / "w_temporal.bin"
    if wt_path.exists():
        result["w_temporal"] = np.fromfile(
            wt_path, dtype=np.float32).reshape((height, width))

    return result


def export_hierarchical_model(checkpoint_path, output_dir):
    """Export a hierarchical checkpoint to binary files for serving/inference.

    Reads the checkpoint directory (metadata.json + .npy weight files) and
    writes flat binary .bin files plus a config.json describing the model.

    Args:
        checkpoint_path: directory containing metadata.json and .npy files
        output_dir: directory to write .bin files and config.json into
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((checkpoint_path / "metadata.json").read_text())
    layer_sizes = metadata["layer_sizes"]
    num_layers = len(layer_sizes)

    # Export prediction weights: pred_weight_i.npy -> pred_weight_i.bin
    for i in range(num_layers - 1):
        src = checkpoint_path / f"pred_weight_{i}.npy"
        if src.exists():
            arr = np.load(src).astype(np.float32)
            arr.tofile(output_dir / f"pred_weight_{i}.bin")

    # Export prediction biases
    for i in range(num_layers - 1):
        src = checkpoint_path / f"pred_bias_{i}.npy"
        if src.exists():
            arr = np.load(src).astype(np.float32)
            arr.tofile(output_dir / f"pred_bias_{i}.bin")

    # Export skip weights
    n_skip = num_layers // 2
    for i in range(n_skip):
        src = checkpoint_path / f"skip_weight_{i}.npy"
        if src.exists():
            arr = np.load(src).astype(np.float32)
            arr.tofile(output_dir / f"skip_weight_{i}.bin")

    # Export skip biases
    for i in range(n_skip):
        src = checkpoint_path / f"skip_bias_{i}.npy"
        if src.exists():
            arr = np.load(src).astype(np.float32)
            arr.tofile(output_dir / f"skip_bias_{i}.bin")

    # Export temporal weights
    for i in range(num_layers):
        src = checkpoint_path / f"temporal_weight_{i}.npy"
        if src.exists():
            arr = np.load(src).astype(np.float32)
            arr.tofile(output_dir / f"temporal_weight_{i}.bin")

    # Export layer representations
    for i in range(num_layers):
        src = checkpoint_path / f"layer_{i}_rep.npy"
        if src.exists():
            arr = np.load(src).astype(np.float32)
            arr.tofile(output_dir / f"layer_{i}_rep.bin")

    # Write config.json
    config = {
        "architecture": "hierarchical",
        "layer_sizes": layer_sizes,
        "num_layers": num_layers,
        "num_instruments": len(metadata.get("vocabulary", {})),
        "vocabulary": metadata.get("vocabulary", {}),
        "conditioning_size": metadata.get("conditioning_size", 0),
        "lr": metadata.get("lr", 0.01),
        "lr_w": metadata.get("lr_w", 0.001),
        "alpha": metadata.get("alpha", 0.8),
        "lambda_sparse": metadata.get("lambda_sparse", 0.005),
        "dtype": "float32",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
