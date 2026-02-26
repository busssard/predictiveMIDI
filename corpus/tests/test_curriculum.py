from corpus.services.curriculum import CurriculumScheduler


class TestCurriculumScheduler:
    def test_starts_at_phase_1(self):
        cs = CurriculumScheduler()
        assert cs.current_phase == 1
        assert cs.snippet_ticks == cs.phases[1]["snippet_ticks"]

    def test_advances_when_error_below_threshold(self):
        cs = CurriculumScheduler()
        phase1_ticks = cs.snippet_ticks
        # Report error above threshold — should stay
        cs.report_error(0.5)
        assert cs.current_phase == 1
        # Report error below threshold enough times
        for _ in range(cs.patience):
            cs.report_error(cs.phases[1]["threshold"] - 0.01)
        assert cs.current_phase == 2
        assert cs.snippet_ticks > phase1_ticks

    def test_does_not_advance_past_final_phase(self):
        cs = CurriculumScheduler()
        max_phase = max(cs.phases.keys())
        cs.current_phase = max_phase
        for _ in range(100):
            cs.report_error(0.001)
        assert cs.current_phase == max_phase

    def test_custom_phases(self):
        phases = {
            1: {"snippet_ticks": 8, "threshold": 0.1},
            2: {"snippet_ticks": 32, "threshold": 0.05},
        }
        cs = CurriculumScheduler(phases=phases, patience=3)
        assert cs.snippet_ticks == 8
