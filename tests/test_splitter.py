"""Tests for the traffic splitter module."""

from datetime import datetime, timedelta

import pytest

from ab_testing_agent.models.experiment import (
    Experiment,
    ExperimentStatus,
    MetricType,
    RampStage,
    RampUpSchedule,
    Variant,
)
from ab_testing_agent.splitter.splitter import TrafficSplitter


def _make_experiment(**kwargs) -> Experiment:
    defaults = dict(
        name="test_exp",
        status=ExperimentStatus.RUNNING,
        primary_metric="conversion",
        primary_metric_type=MetricType.CONVERSION,
        variants=[
            Variant(id="ctrl", name="control", is_control=True, traffic_percentage=50),
            Variant(id="treat", name="treatment", is_control=False, traffic_percentage=50),
        ],
        start_date=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return Experiment(**defaults)


@pytest.fixture
def splitter():
    return TrafficSplitter()


class TestAssignment:
    def test_assigns_to_variant(self, splitter):
        exp = _make_experiment()
        a = splitter.assign_user(exp, "user_1")
        assert a is not None
        assert a.variant_id in ("ctrl", "treat")
        assert a.user_id == "user_1"

    def test_deterministic_assignment(self, splitter):
        exp = _make_experiment()
        a1 = splitter.assign_user(exp, "user_1")
        a2 = splitter.assign_user(exp, "user_1")
        # Second call returns existing assignment
        assert a1.variant_id == a2.variant_id

    def test_different_users_may_get_different_variants(self, splitter):
        exp = _make_experiment()
        variants_seen = set()
        for i in range(100):
            a = splitter.assign_user(exp, f"user_{i}")
            if a:
                variants_seen.add(a.variant_id)
        # With 100 users and 50/50 split, should see both
        assert len(variants_seen) == 2

    def test_no_assignment_when_draft(self, splitter):
        exp = _make_experiment(status=ExperimentStatus.DRAFT)
        a = splitter.assign_user(exp, "user_1")
        assert a is None

    def test_bulk_assign(self, splitter):
        exp = _make_experiment()
        users = [f"user_{i}" for i in range(50)]
        assignments = splitter.bulk_assign(exp, users)
        assert len(assignments) > 0
        assert len(assignments) <= 50


class TestRampUp:
    def test_ramp_stage_detection(self, splitter):
        exp = _make_experiment(
            start_date=datetime.utcnow() - timedelta(hours=1),
            ramp_up_schedule=RampUpSchedule(
                stages=[
                    RampStage(traffic_percent=10, duration_hours=24, description="Stage 1"),
                    RampStage(traffic_percent=50, duration_hours=24, description="Stage 2"),
                    RampStage(traffic_percent=100, duration_hours=24, description="Stage 3"),
                ]
            ),
        )
        stage = splitter.get_current_ramp_stage(exp)
        assert stage is not None
        assert stage.traffic_percent == 10  # 1 hour in, still in first stage

    def test_no_ramp_returns_none(self, splitter):
        exp = _make_experiment()
        stage = splitter.get_current_ramp_stage(exp)
        assert stage is None


class TestLifecycle:
    def test_start_experiment(self, splitter):
        exp = _make_experiment(status=ExperimentStatus.DRAFT, start_date=None)
        exp = splitter.start_experiment(exp)
        assert exp.status == ExperimentStatus.RUNNING
        assert exp.start_date is not None

    def test_stop_experiment(self, splitter):
        exp = _make_experiment()
        exp = splitter.stop_experiment(exp, "test reason")
        assert exp.status == ExperimentStatus.STOPPED_EARLY

    def test_complete_experiment(self, splitter):
        exp = _make_experiment()
        exp = splitter.complete_experiment(exp)
        assert exp.status == ExperimentStatus.COMPLETED

    def test_should_end_by_date(self, splitter):
        exp = _make_experiment(
            end_date=datetime.utcnow() - timedelta(hours=1)
        )
        assert splitter.should_experiment_end(exp) is True

    def test_should_end_by_sample(self, splitter):
        exp = _make_experiment(
            required_sample_size_per_variant=100,
            current_sample_size_per_variant=100,
        )
        assert splitter.should_experiment_end(exp) is True


class TestSchedule:
    def test_build_schedule(self, splitter):
        exp = _make_experiment(
            estimated_duration_days=14,
            ramp_up_schedule=RampUpSchedule(
                stages=[
                    RampStage(traffic_percent=10, duration_hours=24, description="S1"),
                    RampStage(traffic_percent=100, duration_hours=24, description="S2"),
                ]
            ),
        )
        schedule = splitter.build_schedule(exp)
        assert "experiment" in schedule
        assert len(schedule["phases"]) >= 2


class TestAssignmentCounts:
    def test_counts(self, splitter):
        exp = _make_experiment()
        for i in range(20):
            splitter.assign_user(exp, f"u_{i}")
        counts = splitter.get_assignment_counts(exp.id)
        total = sum(counts.values())
        assert total == 20


class TestHashing:
    def test_hash_deterministic(self):
        b1 = TrafficSplitter._hash_to_bucket("exp1", "user1")
        b2 = TrafficSplitter._hash_to_bucket("exp1", "user1")
        assert b1 == b2

    def test_hash_range(self):
        b = TrafficSplitter._hash_to_bucket("exp1", "user1")
        assert 0 <= b < 1

    def test_hash_distribution(self):
        """Check rough uniformity over many users."""
        buckets = [TrafficSplitter._hash_to_bucket("e", f"u_{i}") for i in range(10000)]
        mean = sum(buckets) / len(buckets)
        assert 0.45 < mean < 0.55  # Should be close to 0.5
