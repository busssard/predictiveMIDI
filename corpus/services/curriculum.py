DEFAULT_PHASES = {
    1: {"snippet_ticks": 16, "threshold": 0.15},    # ~2 bars at 8Hz, 120bpm
    2: {"snippet_ticks": 32, "threshold": 0.10},    # ~4 bars
    3: {"snippet_ticks": 64, "threshold": 0.0},     # ~8 bars (final, no advance)
}


class CurriculumScheduler:
    def __init__(self, phases=None, patience=10):
        """
        Args:
            phases: dict mapping phase number to {snippet_ticks, threshold}.
            patience: how many consecutive low-error reports before advancing.
        """
        self.phases = phases or DEFAULT_PHASES
        self.patience = patience
        self.current_phase = 1
        self._consecutive_low = 0

    @property
    def snippet_ticks(self):
        return self.phases[self.current_phase]["snippet_ticks"]

    def report_error(self, avg_error):
        """Report the average error for the latest batch.

        Advances to next phase if error stays below threshold for
        `patience` consecutive reports.
        """
        threshold = self.phases[self.current_phase]["threshold"]
        max_phase = max(self.phases.keys())

        if self.current_phase >= max_phase:
            return

        if avg_error < threshold:
            self._consecutive_low += 1
        else:
            self._consecutive_low = 0

        if self._consecutive_low >= self.patience:
            self.current_phase += 1
            self._consecutive_low = 0
