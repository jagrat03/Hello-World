"""Experiment designer: sample size calculation, power analysis, and test configuration."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from scipy import stats

from ab_testing_agent.models.experiment import (
    Experiment,
    ExperimentStatus,
    GuardrailMetric,
    MetricType,
    RampStage,
    RampUpSchedule,
    Variant,
)


class ExperimentDesigner:
    """Designs A/B tests: computes sample sizes, builds ramp-up plans, validates configs."""

    # ── sample size calculation ──────────────────────────────────────

    @staticmethod
    def required_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80,
        metric_type: MetricType = MetricType.CONVERSION,
        baseline_std: float | None = None,
    ) -> int:
        """Return required sample size *per variant*.

        For conversion metrics uses the normal approximation for two-proportion
        z-test.  For continuous metrics uses the two-sample t-test formula.
        """
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        if metric_type == MetricType.CONVERSION:
            p1 = baseline_rate
            p2 = p1 * (1 + minimum_detectable_effect)
            p2 = min(max(p2, 0.0001), 0.9999)
            pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
            n = ((z_alpha + z_beta) ** 2 * pooled_var) / (p2 - p1) ** 2
        else:
            # Continuous / count / ratio
            if baseline_std is None:
                baseline_std = baseline_rate * 0.5  # rough default
            delta = baseline_rate * minimum_detectable_effect
            if delta == 0:
                return 0
            n = 2 * ((z_alpha + z_beta) * baseline_std / delta) ** 2

        return math.ceil(n)

    # ── power analysis (reverse: what power do we get for a given n?) ──

    @staticmethod
    def compute_power(
        baseline_rate: float,
        minimum_detectable_effect: float,
        sample_size_per_variant: int,
        alpha: float = 0.05,
        metric_type: MetricType = MetricType.CONVERSION,
        baseline_std: float | None = None,
    ) -> float:
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        if metric_type == MetricType.CONVERSION:
            p1 = baseline_rate
            p2 = p1 * (1 + minimum_detectable_effect)
            p2 = min(max(p2, 0.0001), 0.9999)
            se = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / sample_size_per_variant)
        else:
            if baseline_std is None:
                baseline_std = baseline_rate * 0.5
            se = baseline_std * math.sqrt(2 / sample_size_per_variant)

        delta = abs(baseline_rate * minimum_detectable_effect)
        if se == 0:
            return 1.0
        z_beta = delta / se - z_alpha
        return float(stats.norm.cdf(z_beta))

    # ── duration estimation ──────────────────────────────────────────

    @staticmethod
    def estimate_duration_days(
        required_sample_per_variant: int,
        num_variants: int,
        daily_traffic: int,
        traffic_fraction: float = 1.0,
    ) -> float:
        """Estimate how many days the experiment needs to run."""
        total_needed = required_sample_per_variant * num_variants
        daily_allocated = daily_traffic * traffic_fraction
        if daily_allocated <= 0:
            return float("inf")
        return math.ceil(total_needed / daily_allocated)

    # ── ramp-up schedule builder ─────────────────────────────────────

    @staticmethod
    def build_ramp_up_schedule(
        target_traffic_percent: float = 100.0,
        num_stages: int = 3,
        total_ramp_hours: float = 72.0,
    ) -> RampUpSchedule:
        """Create a gradual ramp-up schedule.

        By default ramps 10% -> 50% -> 100% over 72 hours.
        """
        if num_stages < 1:
            num_stages = 1

        hours_per_stage = total_ramp_hours / num_stages
        stages: list[RampStage] = []

        for i in range(num_stages):
            frac = (i + 1) / num_stages
            pct = round(target_traffic_percent * frac, 1)
            stages.append(
                RampStage(
                    traffic_percent=pct,
                    duration_hours=hours_per_stage,
                    description=f"Stage {i + 1}: ramp to {pct}% traffic",
                )
            )

        return RampUpSchedule(stages=stages)

    # ── full experiment design ───────────────────────────────────────

    def design_experiment(
        self,
        name: str,
        hypothesis: str,
        primary_metric: str,
        metric_type: MetricType,
        baseline_rate: float,
        minimum_detectable_effect: float,
        daily_traffic: int,
        variant_names: list[str] | None = None,
        alpha: float = 0.05,
        power: float = 0.80,
        baseline_std: float | None = None,
        guardrail_metrics: list[dict] | None = None,
        ramp_up: bool = True,
        traffic_fraction: float = 1.0,
        owner: str = "",
        tags: list[str] | None = None,
    ) -> Experiment:
        """Design a complete experiment configuration from scratch."""
        if variant_names is None:
            variant_names = ["treatment"]

        num_variants = 1 + len(variant_names)  # +1 for control
        pct_each = round(100.0 / num_variants, 2)

        variants = [
            Variant(name="control", is_control=True, traffic_percentage=pct_each)
        ]
        for vname in variant_names:
            variants.append(Variant(name=vname, traffic_percentage=pct_each))

        sample_size = self.required_sample_size(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=minimum_detectable_effect,
            alpha=alpha,
            power=power,
            metric_type=metric_type,
            baseline_std=baseline_std,
        )

        duration = self.estimate_duration_days(
            required_sample_per_variant=sample_size,
            num_variants=num_variants,
            daily_traffic=daily_traffic,
            traffic_fraction=traffic_fraction,
        )

        ramp_schedule = RampUpSchedule()
        if ramp_up:
            ramp_schedule = self.build_ramp_up_schedule(
                target_traffic_percent=100.0,
                num_stages=3,
                total_ramp_hours=min(72.0, duration * 24 * 0.25),
            )

        guardrails = []
        if guardrail_metrics:
            for gm in guardrail_metrics:
                guardrails.append(GuardrailMetric(**gm))

        experiment = Experiment(
            name=name,
            hypothesis=hypothesis,
            primary_metric=primary_metric,
            primary_metric_type=metric_type,
            minimum_detectable_effect=minimum_detectable_effect,
            significance_level=alpha,
            statistical_power=power,
            variants=variants,
            required_sample_size_per_variant=sample_size,
            daily_traffic_estimate=daily_traffic,
            estimated_duration_days=duration,
            ramp_up_schedule=ramp_schedule,
            guardrail_metrics=guardrails,
            owner=owner,
            tags=tags or [],
            status=ExperimentStatus.DRAFT,
        )

        return experiment

    # ── validation ───────────────────────────────────────────────────

    @staticmethod
    def validate_experiment(experiment: Experiment) -> list[str]:
        """Return a list of validation warnings/errors."""
        issues: list[str] = []

        if not experiment.variants:
            issues.append("Experiment has no variants defined.")
        elif not experiment.get_control():
            issues.append("No control variant marked.")

        total_pct = experiment.total_traffic_allocated()
        if abs(total_pct - 100.0) > 0.5:
            issues.append(
                f"Traffic allocation sums to {total_pct:.1f}%, expected ~100%."
            )

        if experiment.minimum_detectable_effect <= 0:
            issues.append("Minimum detectable effect must be > 0.")

        if (
            experiment.required_sample_size_per_variant
            and experiment.daily_traffic_estimate
        ):
            num_v = len(experiment.variants)
            total = experiment.required_sample_size_per_variant * num_v
            if experiment.daily_traffic_estimate > 0:
                days = total / experiment.daily_traffic_estimate
                if days > 90:
                    issues.append(
                        f"Estimated duration is {days:.0f} days. "
                        "Consider increasing MDE or traffic."
                    )

        return issues
