"""Models for tracking assignments, events, and results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Assignment(BaseModel):
    """Records which variant a user was assigned to."""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Event(BaseModel):
    """A metric event recorded for a user in an experiment."""
    user_id: str
    experiment_id: str
    metric_name: str
    value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VariantResult(BaseModel):
    """Aggregated results for a single variant."""
    variant_id: str
    variant_name: str
    is_control: bool = False
    sample_size: int = 0
    conversions: int = 0
    conversion_rate: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0


class ComparisonResult(BaseModel):
    """Statistical comparison between a treatment variant and control."""
    treatment_variant_id: str
    treatment_variant_name: str
    control_variant_id: str

    # Effect
    absolute_effect: float = 0.0
    relative_effect_percent: float = 0.0

    # Frequentist
    p_value: float = 1.0
    is_significant: bool = False
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    # Bayesian
    probability_of_being_best: float = 0.0
    expected_loss: float = 0.0
    credible_interval: tuple[float, float] = (0.0, 0.0)

    model_config = {"arbitrary_types_allowed": True}


class GuardrailResult(BaseModel):
    """Result of guardrail metric check."""
    metric_name: str
    control_value: float
    treatment_value: float
    degradation_percent: float
    threshold_percent: float
    passed: bool


class ExperimentResults(BaseModel):
    """Full analysis results for an experiment."""
    experiment_id: str
    experiment_name: str
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "in_progress"

    # Per-variant stats
    variant_results: list[VariantResult] = Field(default_factory=list)

    # Treatment vs control comparisons
    comparisons: list[ComparisonResult] = Field(default_factory=list)

    # Guardrail checks
    guardrail_results: list[GuardrailResult] = Field(default_factory=list)
    all_guardrails_passed: bool = True

    # Recommendation
    recommendation: str = ""
    confidence_in_recommendation: str = ""  # "high", "medium", "low"

    # Summaries
    summary: str = ""
    detailed_analysis: str = ""

    # Sequential testing
    can_stop_early: bool = False
    early_stop_reason: str = ""
