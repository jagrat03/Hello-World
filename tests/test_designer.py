"""Tests for the experiment designer module."""

import math

import pytest

from ab_testing_agent.designer.designer import ExperimentDesigner
from ab_testing_agent.models.experiment import (
    Experiment,
    ExperimentStatus,
    MetricType,
    Variant,
)


@pytest.fixture
def designer():
    return ExperimentDesigner()


class TestSampleSize:
    def test_conversion_metric_basic(self, designer):
        n = designer.required_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.05,
            alpha=0.05,
            power=0.80,
            metric_type=MetricType.CONVERSION,
        )
        assert n > 0
        assert isinstance(n, int)
        # For a 10% baseline with 5% relative MDE, should need thousands
        assert n > 1000

    def test_higher_power_needs_more_samples(self, designer):
        n_80 = designer.required_sample_size(0.10, 0.05, power=0.80)
        n_95 = designer.required_sample_size(0.10, 0.05, power=0.95)
        assert n_95 > n_80

    def test_larger_mde_needs_fewer_samples(self, designer):
        n_small = designer.required_sample_size(0.10, 0.10)
        n_large = designer.required_sample_size(0.10, 0.02)
        assert n_large > n_small

    def test_continuous_metric(self, designer):
        n = designer.required_sample_size(
            baseline_rate=50.0,
            minimum_detectable_effect=0.05,
            metric_type=MetricType.CONTINUOUS,
            baseline_std=15.0,
        )
        assert n > 0

    def test_zero_mde_returns_zero(self, designer):
        n = designer.required_sample_size(
            baseline_rate=50.0,
            minimum_detectable_effect=0.0,
            metric_type=MetricType.CONTINUOUS,
            baseline_std=15.0,
        )
        assert n == 0


class TestPowerAnalysis:
    def test_power_increases_with_sample_size(self, designer):
        p1 = designer.compute_power(0.10, 0.05, sample_size_per_variant=1000)
        p2 = designer.compute_power(0.10, 0.05, sample_size_per_variant=50000)
        assert p2 > p1

    def test_power_at_required_sample_is_target(self, designer):
        n = designer.required_sample_size(0.10, 0.05, power=0.80)
        p = designer.compute_power(0.10, 0.05, sample_size_per_variant=n)
        assert abs(p - 0.80) < 0.02  # Should be close to 80%


class TestDurationEstimation:
    def test_basic_duration(self, designer):
        days = designer.estimate_duration_days(
            required_sample_per_variant=10000,
            num_variants=2,
            daily_traffic=5000,
        )
        # 20000 total / 5000 daily = 4 days
        assert days == 4

    def test_traffic_fraction(self, designer):
        days = designer.estimate_duration_days(
            required_sample_per_variant=10000,
            num_variants=2,
            daily_traffic=5000,
            traffic_fraction=0.5,
        )
        assert days == 8

    def test_zero_traffic_returns_inf(self, designer):
        days = designer.estimate_duration_days(10000, 2, 0)
        assert days == float("inf")


class TestRampUpSchedule:
    def test_default_schedule(self, designer):
        schedule = designer.build_ramp_up_schedule()
        assert len(schedule.stages) == 3
        assert schedule.stages[-1].traffic_percent == 100.0

    def test_custom_stages(self, designer):
        schedule = designer.build_ramp_up_schedule(
            target_traffic_percent=80.0,
            num_stages=4,
            total_ramp_hours=48.0,
        )
        assert len(schedule.stages) == 4
        assert schedule.stages[-1].traffic_percent == 80.0
        assert all(s.duration_hours == 12.0 for s in schedule.stages)


class TestDesignExperiment:
    def test_full_design(self, designer):
        exp = designer.design_experiment(
            name="test_exp",
            hypothesis="Treatment works",
            primary_metric="conversion",
            metric_type=MetricType.CONVERSION,
            baseline_rate=0.10,
            minimum_detectable_effect=0.05,
            daily_traffic=10000,
        )

        assert exp.name == "test_exp"
        assert exp.status == ExperimentStatus.DRAFT
        assert len(exp.variants) == 2  # control + 1 treatment
        assert exp.get_control() is not None
        assert exp.required_sample_size_per_variant > 0
        assert exp.estimated_duration_days > 0

    def test_multiple_variants(self, designer):
        exp = designer.design_experiment(
            name="multi",
            hypothesis="Test",
            primary_metric="revenue",
            metric_type=MetricType.CONTINUOUS,
            baseline_rate=50.0,
            minimum_detectable_effect=0.05,
            daily_traffic=20000,
            variant_names=["low_price", "mid_price", "high_price"],
        )

        assert len(exp.variants) == 4  # control + 3
        assert len(exp.get_treatment_variants()) == 3

    def test_traffic_allocation(self, designer):
        exp = designer.design_experiment(
            name="test",
            hypothesis="T",
            primary_metric="c",
            metric_type=MetricType.CONVERSION,
            baseline_rate=0.1,
            minimum_detectable_effect=0.05,
            daily_traffic=10000,
        )
        total = exp.total_traffic_allocated()
        assert abs(total - 100.0) < 1.0


class TestValidation:
    def test_valid_experiment_no_issues(self, designer):
        exp = designer.design_experiment(
            name="valid",
            hypothesis="T",
            primary_metric="c",
            metric_type=MetricType.CONVERSION,
            baseline_rate=0.1,
            minimum_detectable_effect=0.05,
            daily_traffic=10000,
        )
        issues = designer.validate_experiment(exp)
        assert len(issues) == 0

    def test_no_variants_flagged(self, designer):
        exp = Experiment(name="empty")
        issues = designer.validate_experiment(exp)
        assert any("no variants" in i.lower() for i in issues)

    def test_no_control_flagged(self, designer):
        exp = Experiment(
            name="no_ctrl",
            variants=[Variant(name="a", traffic_percentage=100, is_control=False)],
        )
        issues = designer.validate_experiment(exp)
        assert any("control" in i.lower() for i in issues)
