"""AI Agent orchestrator for end-to-end A/B testing.

The agent can:
1. Design experiments from natural language descriptions
2. Validate and refine experiment configs
3. Schedule and manage traffic splits
4. Run statistical analysis and generate recommendations
5. Simulate experiment data for testing

Uses OpenAI-compatible LLM for natural language understanding, falls back to
rule-based logic when no API key is configured.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np

from ab_testing_agent.analyzer.analyzer import ExperimentAnalyzer
from ab_testing_agent.designer.designer import ExperimentDesigner
from ab_testing_agent.models.events import Event, ExperimentResults
from ab_testing_agent.models.experiment import (
    Experiment,
    ExperimentStatus,
    MetricType,
)
from ab_testing_agent.splitter.splitter import TrafficSplitter

# Optional LLM support
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment, misc]


SYSTEM_PROMPT = """\
You are an expert A/B testing agent. You help users design, run, and analyze
A/B tests. You understand statistics, experiment design, and product
experimentation best practices.

When designing experiments, extract these parameters from the user's description:
- name: short experiment name
- hypothesis: what we're testing
- primary_metric: the key metric to measure
- metric_type: one of "conversion", "continuous", "count", "ratio"
- baseline_rate: current metric value (estimate if needed)
- minimum_detectable_effect: smallest relative change worth detecting (as decimal, e.g. 0.05 for 5%)
- daily_traffic: estimated daily users/events
- variant_names: list of treatment variant names
- guardrail_metrics: metrics that must not degrade (list of {name, metric_type, max_degradation_percent})

Respond with valid JSON matching the schema above. If the user hasn't provided
enough info, ask clarifying questions as plain text (not JSON).
"""


class ABTestingAgent:
    """End-to-end AI agent for A/B testing."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
    ):
        self.designer = ExperimentDesigner()
        self.splitter = TrafficSplitter()
        self.analyzer = ExperimentAnalyzer()

        self._experiments: dict[str, Experiment] = {}
        self._events: dict[str, list[Event]] = {}
        self._guardrail_events: dict[str, dict[str, list[Event]]] = {}

        # LLM client (optional)
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._llm = None
        self._model = openai_model
        if api_key and OpenAI is not None:
            self._llm = OpenAI(api_key=api_key)

    # ── high-level agent actions ─────────────────────────────────────

    def handle_message(self, user_message: str) -> str:
        """Process a user message and return the agent's response.

        Dispatches to the appropriate action based on message content.
        """
        msg_lower = user_message.lower().strip()

        # Direct commands
        if msg_lower.startswith("design ") or msg_lower.startswith("create "):
            return self._handle_design(user_message)
        elif msg_lower.startswith("analyze ") or msg_lower.startswith("results "):
            return self._handle_analyze(user_message)
        elif msg_lower.startswith("start "):
            return self._handle_start(user_message)
        elif msg_lower.startswith("stop "):
            return self._handle_stop(user_message)
        elif msg_lower.startswith("status"):
            return self._handle_status(user_message)
        elif msg_lower.startswith("schedule "):
            return self._handle_schedule(user_message)
        elif msg_lower.startswith("simulate "):
            return self._handle_simulate(user_message)
        elif msg_lower.startswith("list"):
            return self._handle_list()
        elif msg_lower.startswith("help"):
            return self._help_text()
        else:
            # Try LLM for freeform input
            return self._handle_freeform(user_message)

    # ── design ───────────────────────────────────────────────────────

    def _handle_design(self, message: str) -> str:
        params = self._extract_design_params(message)
        if isinstance(params, str):
            return params  # Clarifying question or error

        experiment = self.designer.design_experiment(**params)
        issues = self.designer.validate_experiment(experiment)
        self._experiments[experiment.id] = experiment

        lines = [
            f"Experiment designed: {experiment.name}",
            f"  ID: {experiment.id}",
            f"  Hypothesis: {experiment.hypothesis}",
            f"  Primary metric: {experiment.primary_metric} ({experiment.primary_metric_type.value})",
            f"  MDE: {experiment.minimum_detectable_effect:.2%}",
            f"  Required sample/variant: {experiment.required_sample_size_per_variant:,}",
            f"  Estimated duration: {experiment.estimated_duration_days:.0f} days",
            f"  Variants: {', '.join(v.name + (' (control)' if v.is_control else '') for v in experiment.variants)}",
            f"  Traffic split: {', '.join(f'{v.name}={v.traffic_percentage:.1f}%' for v in experiment.variants)}",
        ]

        if experiment.ramp_up_schedule.stages:
            lines.append("  Ramp-up schedule:")
            for stage in experiment.ramp_up_schedule.stages:
                lines.append(f"    {stage.description} ({stage.duration_hours:.0f}h)")

        if experiment.guardrail_metrics:
            lines.append(
                f"  Guardrails: {', '.join(g.name for g in experiment.guardrail_metrics)}"
            )

        if issues:
            lines.append("")
            lines.append("Warnings:")
            for issue in issues:
                lines.append(f"  - {issue}")

        power = self.designer.compute_power(
            baseline_rate=params.get("baseline_rate", 0.1),
            minimum_detectable_effect=params.get("minimum_detectable_effect", 0.05),
            sample_size_per_variant=experiment.required_sample_size_per_variant or 0,
            alpha=experiment.significance_level,
            metric_type=experiment.primary_metric_type,
            baseline_std=params.get("baseline_std"),
        )
        lines.append(f"  Statistical power: {power:.1%}")

        return "\n".join(lines)

    def design_from_params(self, **kwargs) -> Experiment:
        """Design an experiment directly from parameters (programmatic API)."""
        experiment = self.designer.design_experiment(**kwargs)
        self._experiments[experiment.id] = experiment
        return experiment

    # ── analysis ─────────────────────────────────────────────────────

    def _handle_analyze(self, message: str) -> str:
        exp = self._find_experiment(message)
        if isinstance(exp, str):
            return exp

        events = self._events.get(exp.id, [])
        if not events:
            return f"No event data for experiment '{exp.name}'. Use 'simulate {exp.name}' to generate test data."

        guardrails = self._guardrail_events.get(exp.id)
        results = self.analyzer.analyze(exp, events, guardrails)
        return results.summary + "\n\n" + results.detailed_analysis

    def analyze_experiment(
        self, experiment_id: str
    ) -> ExperimentResults | str:
        """Programmatic analysis entry point."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return f"Experiment {experiment_id} not found."

        events = self._events.get(exp.id, [])
        guardrails = self._guardrail_events.get(exp.id)
        return self.analyzer.analyze(exp, events, guardrails)

    # ── start / stop / status ────────────────────────────────────────

    def _handle_start(self, message: str) -> str:
        exp = self._find_experiment(message)
        if isinstance(exp, str):
            return exp
        self.splitter.start_experiment(exp)
        schedule = self.splitter.build_schedule(exp)

        lines = [f"Experiment '{exp.name}' started!"]
        for phase in schedule.get("phases", []):
            lines.append(
                f"  {phase['phase']}: {phase['start']} -> {phase['end']} "
                f"({phase['traffic_percent']}% traffic)"
            )
        return "\n".join(lines)

    def _handle_stop(self, message: str) -> str:
        exp = self._find_experiment(message)
        if isinstance(exp, str):
            return exp
        self.splitter.stop_experiment(exp, reason="Manually stopped by user")
        return f"Experiment '{exp.name}' stopped."

    def _handle_status(self, message: str) -> str:
        if not self._experiments:
            return "No experiments. Use 'design ...' to create one."

        lines = ["Active Experiments:", ""]
        for exp in self._experiments.values():
            n_events = len(self._events.get(exp.id, []))
            pct = ""
            if exp.required_sample_size_per_variant and n_events > 0:
                total_needed = exp.required_sample_size_per_variant * len(exp.variants)
                pct = f" ({n_events / total_needed:.0%} of target)"
            lines.append(
                f"  [{exp.status.value.upper()}] {exp.name} (ID: {exp.id})"
                f" | {n_events} events{pct}"
            )
        return "\n".join(lines)

    def _handle_schedule(self, message: str) -> str:
        exp = self._find_experiment(message)
        if isinstance(exp, str):
            return exp

        schedule = self.splitter.build_schedule(exp, start_date=datetime.utcnow())
        lines = [f"Schedule for '{exp.name}':", ""]
        for phase in schedule.get("phases", []):
            lines.append(
                f"  {phase.get('phase', 'Phase')}: "
                f"{phase['start']} -> {phase['end']} "
                f"({phase['traffic_percent']}% traffic)"
            )
        if schedule.get("end"):
            lines.append(f"\n  Estimated completion: {schedule['end']}")
        return "\n".join(lines)

    def _handle_list(self) -> str:
        if not self._experiments:
            return "No experiments created yet."
        lines = ["Experiments:", ""]
        for exp in self._experiments.values():
            lines.append(f"  {exp.name} [{exp.status.value}] (ID: {exp.id})")
        return "\n".join(lines)

    # ── simulation ───────────────────────────────────────────────────

    def _handle_simulate(self, message: str) -> str:
        exp = self._find_experiment(message)
        if isinstance(exp, str):
            return exp
        return self.simulate_data(exp.id)

    def simulate_data(
        self,
        experiment_id: str,
        n_per_variant: int | None = None,
        treatment_effects: dict[str, float] | None = None,
    ) -> str:
        """Generate simulated event data for an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return f"Experiment {experiment_id} not found."

        rng = np.random.default_rng(42)
        n = n_per_variant or exp.required_sample_size_per_variant or 1000
        events: list[Event] = []

        control = exp.get_control()
        baseline = 0.1  # default

        # Try to infer baseline from MDE
        if exp.primary_metric_type == MetricType.CONVERSION:
            baseline = 0.10  # 10% default conversion rate

        effects = treatment_effects or {}

        for variant in exp.variants:
            if variant.is_control:
                rate = baseline
            else:
                # Apply a realistic effect (or user-specified)
                effect = effects.get(variant.name, exp.minimum_detectable_effect)
                rate = baseline * (1 + effect)

            for i in range(n):
                if exp.primary_metric_type == MetricType.CONVERSION:
                    value = float(rng.random() < rate)
                elif exp.primary_metric_type == MetricType.CONTINUOUS:
                    value = float(rng.normal(rate, rate * 0.3))
                elif exp.primary_metric_type == MetricType.COUNT:
                    value = float(rng.poisson(rate))
                else:
                    value = float(rng.normal(rate, rate * 0.2))

                events.append(
                    Event(
                        user_id=f"user_{variant.id}_{i}",
                        experiment_id=exp.id,
                        metric_name=exp.primary_metric,
                        value=value,
                        metadata={"variant_id": variant.id},
                    )
                )

        self._events[exp.id] = events
        exp.current_sample_size_per_variant = n

        # Also simulate guardrail data
        if exp.guardrail_metrics:
            self._guardrail_events[exp.id] = {}
            for gm in exp.guardrail_metrics:
                g_events = []
                for variant in exp.variants:
                    # Guardrails should generally not degrade
                    grate = 0.15
                    for i in range(n):
                        val = float(rng.random() < grate)
                        g_events.append(
                            Event(
                                user_id=f"user_{variant.id}_{i}",
                                experiment_id=exp.id,
                                metric_name=gm.name,
                                value=val,
                                metadata={"variant_id": variant.id},
                            )
                        )
                self._guardrail_events[exp.id][gm.name] = g_events

        return (
            f"Simulated {n:,} events per variant for '{exp.name}' "
            f"({len(exp.variants)} variants, {len(events):,} total events). "
            f"Use 'analyze {exp.name}' to see results."
        )

    # ── LLM integration ─────────────────────────────────────────────

    def _handle_freeform(self, message: str) -> str:
        """Handle freeform natural language via LLM or fallback."""
        if self._llm:
            return self._llm_handle(message)
        return self._rule_based_handle(message)

    def _llm_handle(self, message: str) -> str:
        """Use LLM to interpret the user's message."""
        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            content = response.choices[0].message.content or ""

            # Try to parse as experiment design JSON
            try:
                params = json.loads(content)
                if "name" in params:
                    return self._handle_design(f"design {json.dumps(params)}")
            except (json.JSONDecodeError, TypeError):
                pass

            return content
        except Exception as e:
            return f"LLM error: {e}\n\nFalling back to rule-based mode.\n{self._help_text()}"

    def _rule_based_handle(self, message: str) -> str:
        """Fallback when no LLM is available."""
        return (
            "I can help you with A/B testing! Here's what I can do:\n\n"
            + self._help_text()
            + "\n\nTip: Set OPENAI_API_KEY env var for natural language support."
        )

    # ── parameter extraction ─────────────────────────────────────────

    def _extract_design_params(self, message: str) -> dict[str, Any] | str:
        """Extract experiment design parameters from a message."""
        # Try LLM first
        if self._llm:
            try:
                response = self._llm.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Design this experiment and respond with JSON only:\n\n{message}",
                        },
                    ],
                    temperature=0.1,
                    max_tokens=1500,
                )
                content = response.choices[0].message.content or ""
                # Strip markdown code fences if present
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]
                params = json.loads(content)
                return self._normalize_params(params)
            except Exception:
                pass

        # Fallback: parse structured command
        return self._parse_design_command(message)

    def _parse_design_command(self, message: str) -> dict[str, Any] | str:
        """Parse a structured design command when no LLM is available.

        Expected format:
            design <name> metric=<metric> baseline=<rate> mde=<effect> traffic=<daily>
        """
        parts = message.split()
        if len(parts) < 2:
            return (
                "Usage: design <name> metric=<metric> baseline=<rate> "
                "mde=<effect> traffic=<daily_traffic> [variants=a,b,c]"
            )

        # Remove the "design" keyword
        parts = parts[1:]

        params: dict[str, Any] = {}
        name_parts = []

        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip().lower()
                val = val.strip()

                if key == "metric":
                    params["primary_metric"] = val
                elif key == "type":
                    params["metric_type"] = MetricType(val)
                elif key == "baseline":
                    params["baseline_rate"] = float(val)
                elif key in ("mde", "effect"):
                    params["minimum_detectable_effect"] = float(val)
                elif key in ("traffic", "daily_traffic"):
                    params["daily_traffic"] = int(val)
                elif key == "variants":
                    params["variant_names"] = [v.strip() for v in val.split(",")]
                elif key == "hypothesis":
                    params["hypothesis"] = val
                elif key == "alpha":
                    params["alpha"] = float(val)
                elif key == "power":
                    params["power"] = float(val)
            else:
                name_parts.append(part)

        if name_parts:
            params["name"] = " ".join(name_parts)
        elif "name" not in params:
            params["name"] = "Untitled Experiment"

        # Defaults
        params.setdefault("primary_metric", "conversion_rate")
        params.setdefault("metric_type", MetricType.CONVERSION)
        params.setdefault("baseline_rate", 0.10)
        params.setdefault("minimum_detectable_effect", 0.05)
        params.setdefault("daily_traffic", 10000)
        params.setdefault("hypothesis", "Treatment improves the primary metric.")

        return params

    @staticmethod
    def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
        """Normalize LLM-extracted params to match designer.design_experiment()."""
        if "metric_type" in params:
            if isinstance(params["metric_type"], str):
                params["metric_type"] = MetricType(params["metric_type"])

        # Ensure numeric types
        for key in ("baseline_rate", "minimum_detectable_effect", "alpha", "power"):
            if key in params:
                params[key] = float(params[key])
        if "daily_traffic" in params:
            params["daily_traffic"] = int(params["daily_traffic"])

        return params

    # ── utilities ────────────────────────────────────────────────────

    def _find_experiment(self, message: str) -> Experiment | str:
        """Find experiment by name or ID from a message."""
        msg = message.lower().strip()
        # Strip command prefix
        for prefix in ("analyze ", "results ", "start ", "stop ", "schedule ", "simulate "):
            if msg.startswith(prefix):
                msg = msg[len(prefix):]
                break

        msg = msg.strip()

        # Try exact ID match
        if msg in self._experiments:
            return self._experiments[msg]

        # Try name match (case-insensitive substring)
        for exp in self._experiments.values():
            if msg in exp.name.lower() or exp.name.lower() in msg:
                return exp

        # Try ID prefix
        for eid, exp in self._experiments.items():
            if eid.startswith(msg):
                return exp

        if not self._experiments:
            return "No experiments exist yet. Use 'design ...' to create one first."
        names = ", ".join(f"'{e.name}'" for e in self._experiments.values())
        return f"Experiment not found. Available experiments: {names}"

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> list[Experiment]:
        return list(self._experiments.values())

    @staticmethod
    def _help_text() -> str:
        return """Commands:
  design <name> metric=<m> baseline=<r> mde=<e> traffic=<n> [variants=a,b]
      Design a new experiment with power analysis and sample size calculation.

  start <experiment_name>
      Start running an experiment (begins traffic splitting).

  stop <experiment_name>
      Stop an experiment early.

  schedule <experiment_name>
      View the ramp-up schedule and timeline.

  simulate <experiment_name>
      Generate simulated data for testing the analysis pipeline.

  analyze <experiment_name>
      Run full statistical analysis (frequentist + Bayesian).

  status
      View all experiments and their current state.

  list
      List all experiments.

  help
      Show this help message.

Examples:
  design checkout_flow metric=conversion_rate baseline=0.12 mde=0.05 traffic=50000
  design pricing_test metric=revenue type=continuous baseline=49.99 mde=0.03 traffic=20000 variants=low,mid,high
  simulate checkout_flow
  analyze checkout_flow"""
