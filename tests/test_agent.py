"""Tests for the AI agent orchestrator."""

import pytest

from ab_testing_agent.agent.orchestrator import ABTestingAgent
from ab_testing_agent.models.experiment import MetricType


@pytest.fixture
def agent():
    return ABTestingAgent()


class TestDesign:
    def test_design_via_command(self, agent):
        result = agent.handle_message(
            "design checkout_test metric=conversion_rate baseline=0.12 mde=0.05 traffic=50000"
        )
        assert "checkout_test" in result
        assert "Required sample/variant" in result or "sample" in result.lower()

    def test_design_via_params(self, agent):
        exp = agent.design_from_params(
            name="programmatic_test",
            hypothesis="Test hypothesis",
            primary_metric="revenue",
            metric_type=MetricType.CONTINUOUS,
            baseline_rate=49.99,
            minimum_detectable_effect=0.03,
            daily_traffic=20000,
            variant_names=["low", "high"],
        )
        assert exp.name == "programmatic_test"
        assert len(exp.variants) == 3  # control + 2

    def test_design_with_variants(self, agent):
        result = agent.handle_message(
            "design price_test metric=revenue type=continuous baseline=50 mde=0.05 "
            "traffic=20000 variants=low,mid,high"
        )
        assert "price_test" in result


class TestSimulation:
    def test_simulate_after_design(self, agent):
        agent.handle_message(
            "design sim_test metric=conversion_rate baseline=0.10 mde=0.05 traffic=10000"
        )
        result = agent.handle_message("simulate sim_test")
        assert "Simulated" in result
        assert "events" in result.lower()

    def test_simulate_nonexistent(self, agent):
        result = agent.handle_message("simulate nonexistent")
        assert "not found" in result.lower() or "no experiment" in result.lower()


class TestAnalysis:
    def test_analyze_after_simulate(self, agent):
        agent.handle_message(
            "design analysis_test metric=conversion_rate baseline=0.10 mde=0.05 traffic=10000"
        )
        agent.handle_message("simulate analysis_test")
        result = agent.handle_message("analyze analysis_test")
        assert "analysis_test" in result.lower() or "Analysis" in result

    def test_analyze_no_data(self, agent):
        agent.handle_message(
            "design nodata_test metric=conversion_rate baseline=0.10 mde=0.05 traffic=10000"
        )
        result = agent.handle_message("analyze nodata_test")
        assert "no event" in result.lower() or "simulate" in result.lower()


class TestLifecycle:
    def test_start(self, agent):
        agent.handle_message(
            "design lifecycle_test metric=conversion_rate baseline=0.10 mde=0.05 traffic=10000"
        )
        result = agent.handle_message("start lifecycle_test")
        assert "started" in result.lower()

    def test_stop(self, agent):
        agent.handle_message(
            "design stop_test metric=conversion_rate baseline=0.10 mde=0.05 traffic=10000"
        )
        agent.handle_message("start stop_test")
        result = agent.handle_message("stop stop_test")
        assert "stopped" in result.lower()

    def test_schedule(self, agent):
        agent.handle_message(
            "design sched_test metric=conversion_rate baseline=0.10 mde=0.05 traffic=10000"
        )
        result = agent.handle_message("schedule sched_test")
        assert "schedule" in result.lower() or "traffic" in result.lower()


class TestStatus:
    def test_status_empty(self, agent):
        result = agent.handle_message("status")
        assert "no experiment" in result.lower()

    def test_status_with_experiments(self, agent):
        agent.handle_message(
            "design status_test metric=conversion_rate baseline=0.10 mde=0.05 traffic=10000"
        )
        result = agent.handle_message("status")
        assert "status_test" in result

    def test_list(self, agent):
        agent.handle_message(
            "design list_test metric=conversion_rate baseline=0.10 mde=0.05 traffic=10000"
        )
        result = agent.handle_message("list")
        assert "list_test" in result


class TestHelp:
    def test_help(self, agent):
        result = agent.handle_message("help")
        assert "design" in result.lower()
        assert "analyze" in result.lower()


class TestEndToEnd:
    def test_full_pipeline(self, agent):
        """Full end-to-end: design -> start -> simulate -> analyze."""
        # Design
        result = agent.handle_message(
            "design e2e_test metric=conversion_rate baseline=0.12 mde=0.05 traffic=50000"
        )
        assert "e2e_test" in result

        # Start
        result = agent.handle_message("start e2e_test")
        assert "started" in result.lower()

        # Simulate
        result = agent.handle_message("simulate e2e_test")
        assert "simulated" in result.lower()

        # Analyze
        result = agent.handle_message("analyze e2e_test")
        assert "recommendation" in result.lower() or "Recommendation" in result

    def test_programmatic_pipeline(self, agent):
        """Full pipeline using the programmatic API."""
        exp = agent.design_from_params(
            name="api_test",
            hypothesis="API test",
            primary_metric="conversion",
            metric_type=MetricType.CONVERSION,
            baseline_rate=0.10,
            minimum_detectable_effect=0.05,
            daily_traffic=10000,
        )

        agent.simulate_data(exp.id)
        results = agent.analyze_experiment(exp.id)

        assert not isinstance(results, str)  # Should be ExperimentResults, not error
        assert results.experiment_name == "api_test"
        assert len(results.variant_results) == 2
        assert len(results.comparisons) == 1
        assert len(results.recommendation) > 0
