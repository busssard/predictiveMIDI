import json
import pytest
from training.engine.metrics_logger import MetricsLogger


class TestMetricsLogger:
    def test_log_step_creates_file(self, tmp_path):
        logger = MetricsLogger(str(tmp_path))
        logger.log_step(1, error=0.5)
        assert (tmp_path / "metrics.jsonl").exists()

    def test_log_step_appends(self, tmp_path):
        logger = MetricsLogger(str(tmp_path))
        logger.log_step(1, error=0.5)
        logger.log_step(2, error=0.3)
        lines = (tmp_path / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_load_metrics_returns_all(self, tmp_path):
        logger = MetricsLogger(str(tmp_path))
        logger.log_step(1, error=0.5)
        logger.log_step(2, error=0.3)
        metrics = logger.load_metrics()
        assert len(metrics) == 2
        assert metrics[0]["step"] == 1
        assert metrics[1]["error"] == 0.3

    def test_load_metrics_after_step(self, tmp_path):
        logger = MetricsLogger(str(tmp_path))
        for i in range(10):
            logger.log_step(i, error=1.0 / (i + 1))
        metrics = logger.load_metrics(after_step=5)
        assert all(m["step"] > 5 for m in metrics)

    def test_log_step_includes_timestamp(self, tmp_path):
        logger = MetricsLogger(str(tmp_path))
        logger.log_step(1, error=0.5)
        metrics = logger.load_metrics()
        assert "timestamp" in metrics[0]

    def test_load_metrics_empty_file(self, tmp_path):
        logger = MetricsLogger(str(tmp_path))
        metrics = logger.load_metrics()
        assert metrics == []
