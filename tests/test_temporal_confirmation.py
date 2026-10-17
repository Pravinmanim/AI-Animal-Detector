"""
Unit tests for TemporalConfirmation — probability-based with exponential decay.

Tests cover:
  - Confidence-based recording (float values)
  - Exponential decay weighting
  - Threshold confirmation logic
  - Edge cases, reset, and config validation

Run with:
    python -m pytest tests/test_temporal_confirmation.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.services.temporal_confirmation import TemporalConfirmation


# ─── Helper ───────────────────────────────────────────────────────────────────

def feed(tc: TemporalConfirmation, values: list[float]) -> None:
    """Feed a list of confidence values into the confirmation window."""
    for val in values:
        tc.record(val)


# ─── Basic confidence-based tests ─────────────────────────────────────────────

class TestConfidenceBasedConfirmation:

    def test_all_zeros_no_alert(self):
        tc = TemporalConfirmation(window_size=5, required_hits=2.5)
        feed(tc, [0.0, 0.0, 0.0, 0.0, 0.0])
        assert tc.is_confirmed() is False
        assert tc.hit_count == 0

    def test_high_confidence_sustained_alert(self):
        """Strong detections across all frames should easily confirm."""
        tc = TemporalConfirmation(window_size=5, required_hits=0.5)
        feed(tc, [0.9, 0.85, 0.88, 0.92, 0.87])
        assert tc.is_confirmed() is True

    def test_low_confidence_flicker_no_alert(self):
        """Weak detections spread across frames should NOT confirm with high threshold."""
        tc = TemporalConfirmation(window_size=5, required_hits=2.5, decay_factor=1.0)
        feed(tc, [0.3, 0.0, 0.3, 0.0, 0.3])
        # Equal weights: sum = (0.3+0+0.3+0+0.3)/5 * 5 = 0.18 (weighted sum)
        assert tc.is_confirmed() is False

    def test_strong_recent_detections_confirm(self):
        """Recent high-confidence detections should confirm due to decay weighting."""
        tc = TemporalConfirmation(window_size=5, required_hits=0.3, decay_factor=1.5)
        feed(tc, [0.0, 0.0, 0.0, 0.9, 0.95])
        # Recent frames have much higher weight → weighted sum should exceed threshold
        assert tc.is_confirmed() is True

    def test_old_detections_fade(self):
        """Old detections in the window should contribute less than recent ones."""
        tc = TemporalConfirmation(window_size=5, required_hits=0.5, decay_factor=2.0)
        feed(tc, [0.9, 0.0, 0.0, 0.0, 0.0])
        # The 0.9 is at the oldest position with the smallest weight
        assert tc.weighted_sum < 0.15  # Much less than 0.9 due to decay


# ─── Exponential decay weight tests ──────────────────────────────────────────

class TestExponentialDecayWeights:

    def test_equal_weights_when_decay_is_1(self):
        """With decay_factor=1.0, all frames should be weighted equally."""
        tc = TemporalConfirmation(window_size=5, required_hits=0.5, decay_factor=1.0)
        # All weights should be 1/5 = 0.2
        for w in tc._weights:
            assert abs(w - 0.2) < 1e-6

    def test_weights_sum_to_one(self):
        tc = TemporalConfirmation(window_size=5, required_hits=0.5, decay_factor=1.5)
        assert abs(sum(tc._weights) - 1.0) < 1e-6

    def test_last_weight_is_largest(self):
        tc = TemporalConfirmation(window_size=5, required_hits=0.5, decay_factor=1.5)
        assert tc._weights[-1] > tc._weights[0]
        # Weights should be monotonically increasing
        for i in range(1, len(tc._weights)):
            assert tc._weights[i] >= tc._weights[i - 1]

    def test_weighted_sum_recent_vs_old(self):
        """Same confidence value should produce higher weighted_sum if placed recently."""
        tc_recent = TemporalConfirmation(window_size=5, required_hits=0.5, decay_factor=2.0)
        feed(tc_recent, [0.0, 0.0, 0.0, 0.0, 0.8])

        tc_old = TemporalConfirmation(window_size=5, required_hits=0.5, decay_factor=2.0)
        feed(tc_old, [0.8, 0.0, 0.0, 0.0, 0.0])

        assert tc_recent.weighted_sum > tc_old.weighted_sum


# ─── Window not yet full ──────────────────────────────────────────────────────

class TestWindowNotFull:

    def test_zero_frames_not_confirmed(self):
        tc = TemporalConfirmation(window_size=5, required_hits=2.5)
        assert tc.is_confirmed() is False
        assert tc.total_count == 0

    def test_partial_window_not_confirmed(self):
        tc = TemporalConfirmation(window_size=5, required_hits=0.3)
        feed(tc, [0.9, 0.9])
        assert tc.is_confirmed() is False  # Window not full yet
        assert tc.total_count == 2

    def test_window_fills_exactly(self):
        tc = TemporalConfirmation(window_size=5, required_hits=0.3, decay_factor=1.0)
        feed(tc, [0.9, 0.9, 0.9, 0.9])
        assert tc.is_confirmed() is False  # 4 < 5
        tc.record(0.0)
        assert tc.is_confirmed() is True  # Now full, high sum


# ─── Rolling window behavior ─────────────────────────────────────────────────

class TestRollingWindow:

    def test_confirmation_drops_when_cleared(self):
        tc = TemporalConfirmation(window_size=5, required_hits=0.3, decay_factor=1.0)
        feed(tc, [0.9, 0.9, 0.9, 0.9, 0.9])
        assert tc.is_confirmed() is True
        feed(tc, [0.0, 0.0, 0.0, 0.0, 0.0])
        assert tc.is_confirmed() is False

    def test_hit_count_tracks_nonzero_entries(self):
        tc = TemporalConfirmation(window_size=5, required_hits=0.5)
        feed(tc, [0.9, 0.0, 0.7, 0.0, 0.8])
        assert tc.hit_count == 3  # Three non-zero entries


# ─── Reset ────────────────────────────────────────────────────────────────────

class TestReset:

    def test_reset_clears_confirmation(self):
        tc = TemporalConfirmation(window_size=5, required_hits=0.3, decay_factor=1.0)
        feed(tc, [0.9, 0.9, 0.9, 0.9, 0.9])
        assert tc.is_confirmed() is True
        tc.reset()
        assert tc.is_confirmed() is False
        assert tc.total_count == 0
        assert tc.hit_count == 0

    def test_reset_then_refill(self):
        tc = TemporalConfirmation(window_size=5, required_hits=0.3, decay_factor=1.0)
        feed(tc, [0.9, 0.9, 0.9, 0.9, 0.9])
        tc.reset()
        feed(tc, [0.0, 0.0, 0.0, 0.0, 0.0])
        assert tc.is_confirmed() is False


# ─── Config validation ────────────────────────────────────────────────────────

class TestConfigValidation:

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            TemporalConfirmation(window_size=0, required_hits=1.0)

    def test_required_hits_zero(self):
        with pytest.raises(ValueError):
            TemporalConfirmation(window_size=5, required_hits=0)

    def test_required_hits_negative(self):
        with pytest.raises(ValueError):
            TemporalConfirmation(window_size=5, required_hits=-1.0)

    def test_window_size_1(self):
        tc = TemporalConfirmation(window_size=1, required_hits=0.5, decay_factor=1.0)
        tc.record(0.8)
        assert tc.is_confirmed() is True
        tc.record(0.0)
        assert tc.is_confirmed() is False


# ─── Summary format ───────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_format(self):
        tc = TemporalConfirmation(window_size=5, required_hits=2.5, decay_factor=1.0)
        feed(tc, [0.5, 0.5, 0.5, 0.5, 0.5])
        summary = tc.summary()
        assert "/" in summary
        assert "2.5" in summary  # required_hits should appear


# ─── ClassificationPolicy tests (unchanged) ──────────────────────────────────

class TestClassificationPolicy:

    def setup_method(self):
        from backend.services.classification_policy import (
            ClassificationPolicy, ClassLabel
        )
        self.ClassLabel = ClassLabel
        self.policy = ClassificationPolicy.from_config(
            target_classes=["snake", "monkey", "wild boar"],
            pet_classes=["dog", "cat"],
        )

    def test_dog_is_pet(self):
        assert self.policy.classify("dog") == self.ClassLabel.PET
        assert self.policy.is_pet("Dog") is True

    def test_cat_is_pet(self):
        assert self.policy.classify("cat") == self.ClassLabel.PET

    def test_snake_is_target(self):
        assert self.policy.classify("snake") == self.ClassLabel.TARGET
        assert self.policy.is_target("SNAKE") is True

    def test_monkey_is_target(self):
        assert self.policy.classify("Monkey") == self.ClassLabel.TARGET

    def test_wild_boar_is_target(self):
        assert self.policy.classify("wild boar") == self.ClassLabel.TARGET

    def test_person_is_ignored(self):
        assert self.policy.classify("person") == self.ClassLabel.IGNORED
        assert self.policy.is_target("person") is False
        assert self.policy.is_pet("person") is False

    def test_car_is_ignored(self):
        assert self.policy.classify("car") == self.ClassLabel.IGNORED

    def test_case_insensitive(self):
        assert self.policy.classify("SNAKE") == self.ClassLabel.TARGET
        assert self.policy.classify("DOG") == self.ClassLabel.PET
        assert self.policy.classify("Wild Boar") == self.ClassLabel.TARGET

    def test_unknown_class_is_ignored(self):
        assert self.policy.classify("bear") == self.ClassLabel.IGNORED
        assert self.policy.classify("") == self.ClassLabel.IGNORED
