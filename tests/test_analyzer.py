"""Tests for the statistical analysis engine."""

import numpy as np
import pytest

from ab_testing_agent.analyzer.analyzer import ExperimentAnalyzer
from ab_testing_agent.models.events import Event
from ab_testing_agent.models.experiment import (
    Experiment,
    ExperimentStatus,
    GuardrailMetric,
    MetricType,
    Variant,
)


def _make_experiment(**kwargs) -> Experiment:
    defaults = dict(
        name="test",
        status=ExperimentStatus.RUNNING,
        primary_metric="conversion",
        primary_metric_type=MetricType.CONVERSION,
        significance_level=0.05,
        statistical_power=0.80,
        minimum_detectable_effect=0.05,
        required_sample_size_per_variant=1000,
        variants=[
            Variant(id="ctrl", name="control", is_control=True, traffic_percentage=50),
            Variant(id="treat", name="treatment", is_control=False, traffic_percentage=50),
        ],
    )
    defaults.update(kwargs)
    return Experiment(**defaults)


def _generate_events(
    experiment: Experiment,
    n_per_variant: int = 1000,
    control_rate: float = 0.10,
    treatment_rate: float = 0.12,
    seed: int = 42,
) -> list[Event]:
    rng = np.random.default_rng(seed)
    events = []
    for variant in experiment.variants:
        rate = control_rate if variant.is_control else treatment_rate
        for i in range(n_per_variant):
            if experiment.primary_metric_type == MetricType.CONVERSION:
                value = float(rng.random() < rate)
            else:
                value = float(rng.normal(rate, rate * 0.3))
            events.append(
                Event(
                    user_id=f"u_{variant.id}_{i}",
                    experiment_id=experiment.id,
                    metric_name=experiment.primary_metric,
                    value=value,
                    metadata={"variant_id": variant.id},
                )
            )
    return events


@pytest.fixture
def analyzer():
    return ExperimentAnalyzer()


class TestFullAnalysis:
    def test_basic_analysis(self, analyzer):
        exp = _make_experiment()
        events = _generate_events(exp)
        results = analyzer.analyze(exp, events)

        assert results.experiment_id == exp.id
        assert len(results.variant_results) == 2
        assert len(results.comparisons) == 1

    def test_significant_difference_detected(self, analyzer):
        exp = _make_experiment()
        # Large sample with clear difference
        events = _generate_events(exp, n_per_variant=10000, control_rate=0.10, treatment_rate=0.15)
        results = analyzer.analyze(exp, events)

        assert results.comparisons[0].is_significant
        assert results.comparisons[0].relative_effect_percent > 0

    def test_no_difference_not_significant(self, analyzer):
        exp = _make_experiment()
        # Same rate for both
        events = _generate_events(exp, n_per_variant=1000, control_rate=0.10, treatment_rate=0.10)
        results = analyzer.analyze(exp, events)

        # Should not be significant (p > 0.05 most of the time)
        # We can't guarantee this 100% due to randomness, but with seed=42 it should hold
        assert results.comparisons[0].p_value > 0.01

    def test_summary_generated(self, analyzer):
        exp = _make_experiment()
        events = _generate_events(exp)
        results = analyzer.analyze(exp, events)

        assert len(results.summary) > 0
        assert exp.name in results.summary

    def test_detailed_analysis_generated(self, analyzer):
        exp = _make_experiment()
        events = _generate_events(exp)
        results = analyzer.analyze(exp, events)

        assert len(results.detailed_analysis) > 0
        assert "DETAILED ANALYSIS" in results.detailed_analysis


class TestContinuousMetric:
    def test_continuous_analysis(self, analyzer):
        exp = _make_experiment(
            primary_metric="revenue",
            primary_metric_type=MetricType.CONTINUOUS,
        )
        events = _generate_events(
            exp, n_per_variant=5000, control_rate=50.0, treatment_rate=55.0
        )
        results = analyzer.analyze(exp, events)

        assert len(results.comparisons) == 1
        assert results.comparisons[0].absolute_effect > 0


class TestBayesian:
    def test_bayesian_probability(self, analyzer):
        exp = _make_experiment()
        events = _generate_events(exp, n_per_variant=5000, control_rate=0.10, treatment_rate=0.15)
        results = analyzer.analyze(exp, events)

        # With strong treatment effect, probability of being best should be high
        assert results.comparisons[0].probability_of_being_best > 0.9

    def test_bayesian_credible_interval(self, analyzer):
        exp = _make_experiment()
        events = _generate_events(exp, n_per_variant=5000, control_rate=0.10, treatment_rate=0.15)
        results = analyzer.analyze(exp, events)

        ci = results.comparisons[0].credible_interval
        assert ci[0] < ci[1]  # Lower bound < upper bound
        # CI should contain the true effect (0.05) most of the time
        assert ci[0] > 0  # Lower bound should be positive for clear winner


class TestGuardrails:
    def test_guardrail_passes(self, analyzer):
        exp = _make_experiment(
            guardrail_metrics=[
                GuardrailMetric(name="latency", max_degradation_percent=10.0)
            ],
        )
        events = _generate_events(exp)

        # Guardrail events: same rate for both, large sample to reduce noise
        rng = np.random.default_rng(42)
        guardrail_events = {
            "latency": [
                Event(
                    user_id=f"u_{v.id}_{i}",
                    experiment_id=exp.id,
                    metric_name="latency",
                    value=float(rng.random() < 0.15),
                    metadata={"variant_id": v.id},
                )
                for v in exp.variants
                for i in range(5000)
            ]
        }

        results = analyzer.analyze(exp, events, guardrail_events)
        assert results.all_guardrails_passed

    def test_guardrail_fails(self, analyzer):
        exp = _make_experiment(
            guardrail_metrics=[
                GuardrailMetric(name="error_rate", max_degradation_percent=2.0)
            ],
        )
        events = _generate_events(exp)

        # Treatment has much higher error rate
        guardrail_events = {
            "error_rate": []
        }
        for v in exp.variants:
            rate = 0.01 if v.is_control else 0.10  # 10x worse
            rng = np.random.default_rng(99)
            for i in range(1000):
                guardrail_events["error_rate"].append(
                    Event(
                        user_id=f"u_{v.id}_{i}",
                        experiment_id=exp.id,
                        metric_name="error_rate",
                        value=float(rng.random() < rate),
                        metadata={"variant_id": v.id},
                    )
                )

        results = analyzer.analyze(exp, events, guardrail_events)
        assert not results.all_guardrails_passed


class TestEarlyStopping:
    def test_early_stop_with_strong_signal(self, analyzer):
        exp = _make_experiment(required_sample_size_per_variant=10000)
        # Provide 60% of required samples with huge effect
        events = _generate_events(
            exp, n_per_variant=6000, control_rate=0.10, treatment_rate=0.20
        )
        results = analyzer.analyze(exp, events)

        # Should recommend early stopping with such a large effect at 60% sample
        assert results.can_stop_early

    def test_no_early_stop_with_little_data(self, analyzer):
        exp = _make_experiment(required_sample_size_per_variant=10000)
        # Only 10% of required samples
        events = _generate_events(
            exp, n_per_variant=1000, control_rate=0.10, treatment_rate=0.15
        )
        results = analyzer.analyze(exp, events)

        assert not results.can_stop_early


class TestRecommendation:
    def test_ship_recommendation(self, analyzer):
        exp = _make_experiment()
        events = _generate_events(exp, n_per_variant=10000, control_rate=0.10, treatment_rate=0.15)
        results = analyzer.analyze(exp, events)

        assert "SHIP" in results.recommendation

    def test_keep_running_recommendation(self, analyzer):
        exp = _make_experiment(required_sample_size_per_variant=50000)
        events = _generate_events(exp, n_per_variant=1000, control_rate=0.10, treatment_rate=0.10)
        results = analyzer.analyze(exp, events)

        assert "KEEP RUNNING" in results.recommendation or "NO SIGNIFICANT" in results.recommendation


class TestEdgeCases:
    def test_empty_events(self, analyzer):
        exp = _make_experiment()
        results = analyzer.analyze(exp, [])

        assert len(results.variant_results) == 2
        assert all(vr.sample_size == 0 for vr in results.variant_results)

    def test_single_event(self, analyzer):
        exp = _make_experiment()
        events = [
            Event(
                user_id="u1",
                experiment_id=exp.id,
                metric_name="conversion",
                value=1.0,
                metadata={"variant_id": "ctrl"},
            )
        ]
        results = analyzer.analyze(exp, events)
        assert results is not None

    def test_multivariate_test(self, analyzer):
        exp = _make_experiment(
            variants=[
                Variant(id="ctrl", name="control", is_control=True, traffic_percentage=25),
                Variant(id="t1", name="variant_a", is_control=False, traffic_percentage=25),
                Variant(id="t2", name="variant_b", is_control=False, traffic_percentage=25),
                Variant(id="t3", name="variant_c", is_control=False, traffic_percentage=25),
            ]
        )
        rng = np.random.default_rng(42)
        events = []
        rates = {"ctrl": 0.10, "t1": 0.12, "t2": 0.08, "t3": 0.15}
        for v in exp.variants:
            rate = rates[v.id]
            for i in range(2000):
                events.append(
                    Event(
                        user_id=f"u_{v.id}_{i}",
                        experiment_id=exp.id,
                        metric_name="conversion",
                        value=float(rng.random() < rate),
                        metadata={"variant_id": v.id},
                    )
                )

        results = analyzer.analyze(exp, events)
        assert len(results.comparisons) == 3  # 3 treatments vs control
