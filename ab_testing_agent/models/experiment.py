"""Core data models for A/B test experiments."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"


class MetricType(str, Enum):
    CONVERSION = "conversion"  # binary (0/1)
    CONTINUOUS = "continuous"  # e.g. revenue, time on page
    COUNT = "count"  # e.g. page views per session
    RATIO = "ratio"  # e.g. revenue per visitor


class Variant(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    description: str = ""
    traffic_percentage: float = Field(ge=0, le=100)
    is_control: bool = False
    config: dict[str, Any] = Field(default_factory=dict)


class GuardrailMetric(BaseModel):
    """A metric that must not degrade beyond a threshold to keep the test safe."""
    name: str
    metric_type: MetricType = MetricType.CONVERSION
    max_degradation_percent: float = Field(
        default=5.0,
        description="Max acceptable relative degradation vs control (%)",
    )


class RampUpSchedule(BaseModel):
    """Defines how traffic ramps up over time."""
    model_config = {"arbitrary_types_allowed": True}
    stages: list[RampStage] = Field(default_factory=list)


class RampStage(BaseModel):
    traffic_percent: float = Field(ge=0, le=100)
    duration_hours: float = Field(gt=0)
    description: str = ""


# Re-define RampUpSchedule after RampStage is defined
RampUpSchedule.model_rebuild()


class Experiment(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    hypothesis: str = ""
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Variants
    variants: list[Variant] = Field(default_factory=list)

    # Primary metric
    primary_metric: str = ""
    primary_metric_type: MetricType = MetricType.CONVERSION
    minimum_detectable_effect: float = Field(
        default=0.05,
        description="Minimum relative effect size worth detecting",
    )

    # Statistical parameters
    significance_level: float = Field(default=0.05, ge=0.001, le=0.2)
    statistical_power: float = Field(default=0.8, ge=0.5, le=0.99)

    # Sample size
    required_sample_size_per_variant: int | None = None
    current_sample_size_per_variant: int = 0

    # Scheduling
    start_date: datetime | None = None
    end_date: datetime | None = None
    estimated_duration_days: float | None = None
    daily_traffic_estimate: int | None = None
    ramp_up_schedule: RampUpSchedule = Field(default_factory=RampUpSchedule)

    # Guardrails
    guardrail_metrics: list[GuardrailMetric] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: list[str] = Field(default_factory=list)
    owner: str = ""
    notes: str = ""

    def get_control(self) -> Variant | None:
        for v in self.variants:
            if v.is_control:
                return v
        return None

    def get_treatment_variants(self) -> list[Variant]:
        return [v for v in self.variants if not v.is_control]

    def total_traffic_allocated(self) -> float:
        return sum(v.traffic_percentage for v in self.variants)

    def estimated_end_date(self) -> datetime | None:
        if self.start_date and self.estimated_duration_days:
            return self.start_date + timedelta(days=self.estimated_duration_days)
        return self.end_date
