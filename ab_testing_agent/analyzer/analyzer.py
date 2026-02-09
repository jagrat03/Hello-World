"""Statistical analysis engine for A/B tests.

Supports both frequentist and Bayesian approaches for conversion and
continuous metrics, plus sequential testing (early stopping), guardrail
checks, and automatic recommendation generation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from ab_testing_agent.models.experiment import Experiment, MetricType
from ab_testing_agent.models.events import (
    ComparisonResult,
    Event,
    ExperimentResults,
    GuardrailResult,
    VariantResult,
)


class ExperimentAnalyzer:
    """Runs statistical analysis on A/B test data."""

    # ── public entry point ───────────────────────────────────────────

    def analyze(
        self,
        experiment: Experiment,
        events: list[Event],
        guardrail_events: dict[str, list[Event]] | None = None,
    ) -> ExperimentResults:
        """Full end-to-end analysis of an experiment.

        Parameters
        ----------
        experiment : Experiment
            The experiment config.
        events : list[Event]
            Primary metric events.  Each event has a user_id, variant can be
            inferred from the experiment assignments, and a value.
        guardrail_events : dict[str, list[Event]] | None
            Mapping of guardrail metric name -> events for that metric.
        """
        # Group events by variant
        variant_events = self._group_by_variant(experiment, events)

        # Compute per-variant stats
        variant_results = [
            self._variant_stats(
                variant_id=vid,
                variant_name=self._variant_name(experiment, vid),
                is_control=self._is_control(experiment, vid),
                values=vals,
                metric_type=experiment.primary_metric_type,
                alpha=experiment.significance_level,
            )
            for vid, vals in variant_events.items()
        ]

        # Control variant result
        control_result = next((vr for vr in variant_results if vr.is_control), None)

        # Compare each treatment vs control
        comparisons = []
        if control_result:
            control_vals = variant_events.get(
                self._control_id(experiment), []
            )
            for vr in variant_results:
                if vr.is_control:
                    continue
                treatment_vals = variant_events.get(vr.variant_id, [])
                comp = self._compare(
                    control_result=control_result,
                    treatment_result=vr,
                    control_vals=control_vals,
                    treatment_vals=treatment_vals,
                    metric_type=experiment.primary_metric_type,
                    alpha=experiment.significance_level,
                )
                comparisons.append(comp)

        # Guardrail checks
        guardrail_results = []
        if guardrail_events and experiment.guardrail_metrics:
            guardrail_results = self._check_guardrails(
                experiment, guardrail_events
            )

        all_guardrails_passed = all(g.passed for g in guardrail_results)

        # Sequential testing / early stopping check
        can_stop, stop_reason = self._check_early_stopping(
            experiment, comparisons, variant_results
        )

        # Generate recommendation
        recommendation, confidence = self._generate_recommendation(
            experiment, comparisons, all_guardrails_passed, variant_results
        )

        summary = self._build_summary(
            experiment, variant_results, comparisons, recommendation
        )

        return ExperimentResults(
            experiment_id=experiment.id,
            experiment_name=experiment.name,
            variant_results=variant_results,
            comparisons=comparisons,
            guardrail_results=guardrail_results,
            all_guardrails_passed=all_guardrails_passed,
            can_stop_early=can_stop,
            early_stop_reason=stop_reason,
            recommendation=recommendation,
            confidence_in_recommendation=confidence,
            summary=summary,
            detailed_analysis=self._detailed_report(
                experiment, variant_results, comparisons, guardrail_results
            ),
        )

    # ── per-variant statistics ───────────────────────────────────────

    @staticmethod
    def _variant_stats(
        variant_id: str,
        variant_name: str,
        is_control: bool,
        values: list[float],
        metric_type: MetricType,
        alpha: float,
    ) -> VariantResult:
        n = len(values)
        if n == 0:
            return VariantResult(
                variant_id=variant_id,
                variant_name=variant_name,
                is_control=is_control,
            )

        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

        # Confidence interval
        if n > 1:
            se = std / math.sqrt(n)
            t_crit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci_lower = mean - t_crit * se
            ci_upper = mean + t_crit * se
        else:
            ci_lower = ci_upper = mean

        conversions = int(np.sum(arr)) if metric_type == MetricType.CONVERSION else 0
        conv_rate = mean if metric_type == MetricType.CONVERSION else 0.0

        return VariantResult(
            variant_id=variant_id,
            variant_name=variant_name,
            is_control=is_control,
            sample_size=n,
            conversions=conversions,
            conversion_rate=conv_rate,
            mean=mean,
            std_dev=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    # ── treatment vs control comparison ──────────────────────────────

    def _compare(
        self,
        control_result: VariantResult,
        treatment_result: VariantResult,
        control_vals: list[float],
        treatment_vals: list[float],
        metric_type: MetricType,
        alpha: float,
    ) -> ComparisonResult:
        abs_effect = treatment_result.mean - control_result.mean
        rel_effect = (
            (abs_effect / control_result.mean * 100)
            if control_result.mean != 0
            else 0.0
        )

        # Frequentist test
        p_value, ci = self._frequentist_test(
            control_vals, treatment_vals, metric_type, alpha
        )
        is_sig = p_value < alpha

        # Bayesian analysis
        prob_best, expected_loss, credible = self._bayesian_test(
            control_vals, treatment_vals, metric_type
        )

        return ComparisonResult(
            treatment_variant_id=treatment_result.variant_id,
            treatment_variant_name=treatment_result.variant_name,
            control_variant_id=control_result.variant_id,
            absolute_effect=abs_effect,
            relative_effect_percent=rel_effect,
            p_value=p_value,
            is_significant=is_sig,
            confidence_interval=ci,
            probability_of_being_best=prob_best,
            expected_loss=expected_loss,
            credible_interval=credible,
        )

    # ── frequentist tests ────────────────────────────────────────────

    @staticmethod
    def _frequentist_test(
        control_vals: list[float],
        treatment_vals: list[float],
        metric_type: MetricType,
        alpha: float,
    ) -> tuple[float, tuple[float, float]]:
        """Return (p_value, confidence_interval_of_difference)."""
        if len(control_vals) < 2 or len(treatment_vals) < 2:
            return 1.0, (0.0, 0.0)

        c = np.array(control_vals, dtype=float)
        t = np.array(treatment_vals, dtype=float)

        if metric_type == MetricType.CONVERSION:
            # Two-proportion z-test
            n_c, n_t = len(c), len(t)
            p_c = np.mean(c)
            p_t = np.mean(t)
            p_pool = (np.sum(c) + np.sum(t)) / (n_c + n_t)

            se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            if se == 0:
                return 1.0, (0.0, 0.0)

            z = (p_t - p_c) / se
            p_value = float(2 * (1 - sp_stats.norm.cdf(abs(z))))

            # CI of difference
            se_diff = math.sqrt(
                p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t
            )
            z_crit = sp_stats.norm.ppf(1 - alpha / 2)
            diff = p_t - p_c
            ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)
        else:
            # Welch's t-test
            stat, p_value = sp_stats.ttest_ind(t, c, equal_var=False)
            p_value = float(p_value)

            # CI of difference in means
            diff = float(np.mean(t) - np.mean(c))
            se_diff = math.sqrt(
                float(np.var(t, ddof=1)) / len(t)
                + float(np.var(c, ddof=1)) / len(c)
            )
            # Welch-Satterthwaite degrees of freedom
            v_t = float(np.var(t, ddof=1)) / len(t)
            v_c = float(np.var(c, ddof=1)) / len(c)
            if v_t + v_c == 0:
                return 1.0, (0.0, 0.0)
            df = (v_t + v_c) ** 2 / (
                v_t ** 2 / (len(t) - 1) + v_c ** 2 / (len(c) - 1)
            )
            t_crit = sp_stats.t.ppf(1 - alpha / 2, df=df)
            ci = (diff - t_crit * se_diff, diff + t_crit * se_diff)

        return p_value, (float(ci[0]), float(ci[1]))

    # ── bayesian analysis ────────────────────────────────────────────

    @staticmethod
    def _bayesian_test(
        control_vals: list[float],
        treatment_vals: list[float],
        metric_type: MetricType,
        n_samples: int = 50_000,
    ) -> tuple[float, float, tuple[float, float]]:
        """Bayesian A/B test using Monte Carlo simulation.

        Returns (prob_treatment_is_best, expected_loss, 95% credible_interval).
        """
        if len(control_vals) < 1 or len(treatment_vals) < 1:
            return 0.5, 0.0, (0.0, 0.0)

        rng = np.random.default_rng(42)

        if metric_type == MetricType.CONVERSION:
            # Beta-Binomial model with uniform prior Beta(1, 1)
            c = np.array(control_vals)
            t = np.array(treatment_vals)
            alpha_c = 1 + np.sum(c)
            beta_c = 1 + len(c) - np.sum(c)
            alpha_t = 1 + np.sum(t)
            beta_t = 1 + len(t) - np.sum(t)

            samples_c = rng.beta(alpha_c, beta_c, size=n_samples)
            samples_t = rng.beta(alpha_t, beta_t, size=n_samples)
        else:
            # Normal-Normal model (conjugate with known-ish variance)
            c = np.array(control_vals, dtype=float)
            t = np.array(treatment_vals, dtype=float)

            # Posterior for mean: Normal(sample_mean, se^2)
            mean_c = float(np.mean(c))
            se_c = float(np.std(c, ddof=1) / math.sqrt(len(c))) if len(c) > 1 else 1.0
            mean_t = float(np.mean(t))
            se_t = float(np.std(t, ddof=1) / math.sqrt(len(t))) if len(t) > 1 else 1.0

            samples_c = rng.normal(mean_c, se_c, size=n_samples)
            samples_t = rng.normal(mean_t, se_t, size=n_samples)

        # Probability treatment is best
        prob_best = float(np.mean(samples_t > samples_c))

        # Expected loss (if we choose treatment but control is actually better)
        loss = np.maximum(samples_c - samples_t, 0)
        expected_loss = float(np.mean(loss))

        # 95% credible interval of difference
        diff_samples = samples_t - samples_c
        ci_lower = float(np.percentile(diff_samples, 2.5))
        ci_upper = float(np.percentile(diff_samples, 97.5))

        return prob_best, expected_loss, (ci_lower, ci_upper)

    # ── guardrail checks ─────────────────────────────────────────────

    def _check_guardrails(
        self,
        experiment: Experiment,
        guardrail_events: dict[str, list[Event]],
    ) -> list[GuardrailResult]:
        results = []
        control_id = self._control_id(experiment)

        for gm in experiment.guardrail_metrics:
            events = guardrail_events.get(gm.name, [])
            if not events:
                continue

            control_vals = [e.value for e in events if self._event_variant(experiment, e) == control_id]
            treatment_vals = [e.value for e in events if self._event_variant(experiment, e) != control_id]

            if not control_vals or not treatment_vals:
                continue

            control_mean = float(np.mean(control_vals))
            treatment_mean = float(np.mean(treatment_vals))

            if control_mean != 0:
                degradation = abs((control_mean - treatment_mean) / control_mean) * 100
            else:
                degradation = 0.0 if treatment_mean == 0 else 100.0

            results.append(
                GuardrailResult(
                    metric_name=gm.name,
                    control_value=control_mean,
                    treatment_value=treatment_mean,
                    degradation_percent=degradation,
                    threshold_percent=gm.max_degradation_percent,
                    passed=degradation <= gm.max_degradation_percent,
                )
            )

        return results

    # ── sequential testing / early stopping ──────────────────────────

    @staticmethod
    def _check_early_stopping(
        experiment: Experiment,
        comparisons: list[ComparisonResult],
        variant_results: list[VariantResult],
    ) -> tuple[bool, str]:
        """Use an O'Brien-Fleming-like spending function for early stopping.

        Simplified: we allow stopping if we've hit at least 50% of the
        required sample and the evidence is overwhelming (p < alpha/5).
        """
        if not experiment.required_sample_size_per_variant:
            return False, ""

        control_vr = next((vr for vr in variant_results if vr.is_control), None)
        if not control_vr:
            return False, ""

        fraction_complete = (
            control_vr.sample_size / experiment.required_sample_size_per_variant
        )
        if fraction_complete < 0.5:
            return False, ""

        # Adjusted alpha (spending function)
        adjusted_alpha = experiment.significance_level / max(
            1, 5 * (1 - fraction_complete)
        )

        for comp in comparisons:
            if comp.p_value < adjusted_alpha:
                return True, (
                    f"Strong evidence for {comp.treatment_variant_name} "
                    f"(p={comp.p_value:.6f} < {adjusted_alpha:.6f}) "
                    f"at {fraction_complete:.0%} of required sample."
                )

        # Also allow stopping for futility (very high p-value past 80%)
        if fraction_complete >= 0.8:
            all_futile = all(c.p_value > 0.5 for c in comparisons)
            if all_futile and comparisons:
                return True, (
                    "Futility: no treatment shows a meaningful signal at "
                    f"{fraction_complete:.0%} of required sample."
                )

        return False, ""

    # ── recommendation engine ────────────────────────────────────────

    @staticmethod
    def _generate_recommendation(
        experiment: Experiment,
        comparisons: list[ComparisonResult],
        guardrails_passed: bool,
        variant_results: list[VariantResult],
    ) -> tuple[str, str]:
        """Generate a recommendation based on statistical results."""
        if not comparisons:
            return "Insufficient data to make a recommendation.", "low"

        # Find best treatment
        best = max(comparisons, key=lambda c: c.relative_effect_percent)

        if not guardrails_passed:
            return (
                f"DO NOT SHIP. Guardrail metrics have been violated. "
                f"Best variant ({best.treatment_variant_name}) showed "
                f"{best.relative_effect_percent:+.2f}% lift but guardrails failed.",
                "high",
            )

        if best.is_significant and best.relative_effect_percent > 0:
            conf = "high" if best.probability_of_being_best > 0.95 else "medium"
            return (
                f"SHIP {best.treatment_variant_name}. "
                f"Statistically significant lift of {best.relative_effect_percent:+.2f}% "
                f"(p={best.p_value:.4f}). "
                f"Bayesian probability of being best: {best.probability_of_being_best:.1%}.",
                conf,
            )

        if best.probability_of_being_best > 0.9:
            return (
                f"LIKELY SHIP {best.treatment_variant_name}. "
                f"Not yet statistically significant (p={best.p_value:.4f}) but "
                f"Bayesian analysis gives {best.probability_of_being_best:.1%} "
                f"probability of being best. Consider running longer.",
                "medium",
            )

        control_vr = next((vr for vr in variant_results if vr.is_control), None)
        if control_vr and experiment.required_sample_size_per_variant:
            pct = control_vr.sample_size / experiment.required_sample_size_per_variant
            if pct < 1.0:
                return (
                    f"KEEP RUNNING. Only {pct:.0%} of required sample collected. "
                    f"Current best is {best.treatment_variant_name} at "
                    f"{best.relative_effect_percent:+.2f}% (p={best.p_value:.4f}).",
                    "low",
                )

        return (
            f"NO SIGNIFICANT DIFFERENCE detected. Consider keeping control. "
            f"Best variant {best.treatment_variant_name} showed "
            f"{best.relative_effect_percent:+.2f}% (p={best.p_value:.4f}).",
            "medium",
        )

    # ── reporting ────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        experiment: Experiment,
        variant_results: list[VariantResult],
        comparisons: list[ComparisonResult],
        recommendation: str,
    ) -> str:
        lines = [
            f"=== {experiment.name} ===",
            f"Primary metric: {experiment.primary_metric} ({experiment.primary_metric_type.value})",
            "",
            "Variant Results:",
        ]
        for vr in variant_results:
            label = " (control)" if vr.is_control else ""
            if experiment.primary_metric_type == MetricType.CONVERSION:
                lines.append(
                    f"  {vr.variant_name}{label}: "
                    f"{vr.conversion_rate:.4f} ({vr.conversions}/{vr.sample_size}) "
                    f"CI: [{vr.ci_lower:.4f}, {vr.ci_upper:.4f}]"
                )
            else:
                lines.append(
                    f"  {vr.variant_name}{label}: "
                    f"mean={vr.mean:.4f} std={vr.std_dev:.4f} n={vr.sample_size} "
                    f"CI: [{vr.ci_lower:.4f}, {vr.ci_upper:.4f}]"
                )

        if comparisons:
            lines.append("")
            lines.append("Comparisons vs Control:")
            for c in comparisons:
                lines.append(
                    f"  {c.treatment_variant_name}: "
                    f"lift={c.relative_effect_percent:+.2f}% "
                    f"p={c.p_value:.4f} "
                    f"{'*** SIGNIFICANT ***' if c.is_significant else 'not significant'} "
                    f"| Bayesian P(best)={c.probability_of_being_best:.1%}"
                )

        lines += ["", f"Recommendation: {recommendation}"]
        return "\n".join(lines)

    @staticmethod
    def _detailed_report(
        experiment: Experiment,
        variant_results: list[VariantResult],
        comparisons: list[ComparisonResult],
        guardrail_results: list[GuardrailResult],
    ) -> str:
        lines = [
            "=" * 60,
            f"DETAILED ANALYSIS: {experiment.name}",
            "=" * 60,
            "",
            f"Hypothesis: {experiment.hypothesis}",
            f"Primary Metric: {experiment.primary_metric}",
            f"Metric Type: {experiment.primary_metric_type.value}",
            f"Significance Level: {experiment.significance_level}",
            f"Required Power: {experiment.statistical_power}",
            f"MDE: {experiment.minimum_detectable_effect:.2%}",
            f"Required Sample Size/Variant: {experiment.required_sample_size_per_variant}",
            "",
        ]

        lines.append("--- VARIANT DETAILS ---")
        for vr in variant_results:
            lines += [
                f"\n{vr.variant_name} ({'CONTROL' if vr.is_control else 'TREATMENT'}):",
                f"  Sample size: {vr.sample_size}",
                f"  Mean: {vr.mean:.6f}",
                f"  Std Dev: {vr.std_dev:.6f}",
                f"  95% CI: [{vr.ci_lower:.6f}, {vr.ci_upper:.6f}]",
            ]
            if experiment.primary_metric_type == MetricType.CONVERSION:
                lines.append(
                    f"  Conversions: {vr.conversions}/{vr.sample_size} "
                    f"({vr.conversion_rate:.4%})"
                )

        if comparisons:
            lines += ["", "--- STATISTICAL COMPARISONS ---"]
            for c in comparisons:
                lines += [
                    f"\n{c.treatment_variant_name} vs Control:",
                    f"  Absolute Effect: {c.absolute_effect:+.6f}",
                    f"  Relative Effect: {c.relative_effect_percent:+.2f}%",
                    "",
                    "  Frequentist:",
                    f"    p-value: {c.p_value:.6f}",
                    f"    Significant: {'YES' if c.is_significant else 'NO'}",
                    f"    95% CI of difference: [{c.confidence_interval[0]:.6f}, {c.confidence_interval[1]:.6f}]",
                    "",
                    "  Bayesian:",
                    f"    P(treatment is best): {c.probability_of_being_best:.2%}",
                    f"    Expected loss: {c.expected_loss:.6f}",
                    f"    95% Credible interval: [{c.credible_interval[0]:.6f}, {c.credible_interval[1]:.6f}]",
                ]

        if guardrail_results:
            lines += ["", "--- GUARDRAIL CHECKS ---"]
            for g in guardrail_results:
                status = "PASSED" if g.passed else "FAILED"
                lines.append(
                    f"  {g.metric_name}: {status} "
                    f"(degradation={g.degradation_percent:.2f}%, "
                    f"threshold={g.threshold_percent:.2f}%)"
                )

        return "\n".join(lines)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _group_by_variant(
        experiment: Experiment, events: list[Event]
    ) -> dict[str, list[float]]:
        """Group event values by variant ID using the event metadata."""
        groups: dict[str, list[float]] = {v.id: [] for v in experiment.variants}
        for event in events:
            vid = event.metadata.get("variant_id", "")
            if vid in groups:
                groups[vid].append(event.value)
        return groups

    @staticmethod
    def _variant_name(experiment: Experiment, variant_id: str) -> str:
        for v in experiment.variants:
            if v.id == variant_id:
                return v.name
        return variant_id

    @staticmethod
    def _is_control(experiment: Experiment, variant_id: str) -> bool:
        for v in experiment.variants:
            if v.id == variant_id:
                return v.is_control
        return False

    @staticmethod
    def _control_id(experiment: Experiment) -> str:
        for v in experiment.variants:
            if v.is_control:
                return v.id
        return ""

    @staticmethod
    def _event_variant(experiment: Experiment, event: Event) -> str:
        return event.metadata.get("variant_id", "")
