import json
import time
from pathlib import Path


class MetricsLogger:
    def __init__(self, log_dir: str):
        self.log_path = Path(log_dir) / "metrics.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_step(self, step: int, **metrics):
        record = {"step": step, "timestamp": time.time(), **metrics}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_metrics(self, after_step=0):
        if not self.log_path.exists():
            return []
        metrics = []
        for line in self.log_path.read_text().strip().split("\n"):
            if line:
                record = json.loads(line)
                if record["step"] > after_step:
                    metrics.append(record)
        return metrics
