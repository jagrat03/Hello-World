"""Traffic splitter: assigns users to variants and manages ramp-up scheduling."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from ab_testing_agent.models.experiment import Experiment, ExperimentStatus, RampStage
from ab_testing_agent.models.events import Assignment


class TrafficSplitter:
    """Deterministic, hash-based traffic splitter with ramp-up scheduling."""

    def __init__(self, storage_path: str | Path | None = None):
        self._storage_path = Path(storage_path) if storage_path else None
        self._assignments: dict[str, list[Assignment]] = {}  # experiment_id -> list
        self._experiments: dict[str, Experiment] = {}

    # ── assignment ───────────────────────────────────────────────────

    def assign_user(
        self,
        experiment: Experiment,
        user_id: str,
        attributes: dict | None = None,
    ) -> Assignment | None:
        """Assign a user to a variant deterministically using hashing.

        Returns None if the experiment isn't running or user falls outside
        the current traffic allocation (during ramp-up).
        """
        if experiment.status not in (
            ExperimentStatus.RUNNING,
            ExperimentStatus.SCHEDULED,
        ):
            return None

        # Check if user already assigned
        existing = self._find_existing_assignment(experiment.id, user_id)
        if existing:
            return existing

        # Determine current traffic fraction from ramp-up schedule
        traffic_fraction = self._current_traffic_fraction(experiment)

        # Hash user into [0, 1) range
        bucket = self._hash_to_bucket(experiment.id, user_id)

        # If outside current ramp-up window, skip
        if bucket >= traffic_fraction:
            return None

        # Assign to variant based on traffic percentages
        variant = self._select_variant(experiment, bucket, traffic_fraction)
        if variant is None:
            return None

        assignment = Assignment(
            user_id=user_id,
            experiment_id=experiment.id,
            variant_id=variant.id,
            attributes=attributes or {},
        )

        self._assignments.setdefault(experiment.id, []).append(assignment)
        return assignment

    def bulk_assign(
        self,
        experiment: Experiment,
        user_ids: list[str],
    ) -> list[Assignment]:
        """Assign multiple users at once."""
        results = []
        for uid in user_ids:
            a = self.assign_user(experiment, uid)
            if a:
                results.append(a)
        return results

    # ── scheduling ───────────────────────────────────────────────────

    def get_current_ramp_stage(self, experiment: Experiment) -> RampStage | None:
        """Return which ramp-up stage the experiment is currently in."""
        if not experiment.ramp_up_schedule.stages or not experiment.start_date:
            return None

        elapsed_hours = (
            datetime.utcnow() - experiment.start_date
        ).total_seconds() / 3600.0
        cumulative = 0.0

        for stage in experiment.ramp_up_schedule.stages:
            cumulative += stage.duration_hours
            if elapsed_hours <= cumulative:
                return stage

        # Past all stages - return the last one (full traffic)
        return experiment.ramp_up_schedule.stages[-1]

    def start_experiment(self, experiment: Experiment) -> Experiment:
        """Transition experiment to RUNNING status."""
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.utcnow()
        if experiment.estimated_duration_days:
            experiment.end_date = experiment.start_date + timedelta(
                days=experiment.estimated_duration_days
            )
        experiment.updated_at = datetime.utcnow()
        self._experiments[experiment.id] = experiment
        return experiment

    def pause_experiment(self, experiment: Experiment) -> Experiment:
        experiment.status = ExperimentStatus.PAUSED
        experiment.updated_at = datetime.utcnow()
        return experiment

    def stop_experiment(self, experiment: Experiment, reason: str = "") -> Experiment:
        experiment.status = ExperimentStatus.STOPPED_EARLY
        experiment.updated_at = datetime.utcnow()
        experiment.notes += f"\nStopped early: {reason}" if reason else ""
        return experiment

    def complete_experiment(self, experiment: Experiment) -> Experiment:
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.utcnow()
        experiment.updated_at = datetime.utcnow()
        return experiment

    def should_experiment_end(self, experiment: Experiment) -> bool:
        """Check if experiment has reached its planned end date or sample size."""
        if experiment.end_date and datetime.utcnow() >= experiment.end_date:
            return True
        if (
            experiment.required_sample_size_per_variant
            and experiment.current_sample_size_per_variant
            >= experiment.required_sample_size_per_variant
        ):
            return True
        return False

    # ── schedule builder ─────────────────────────────────────────────

    def build_schedule(
        self,
        experiment: Experiment,
        start_date: datetime | None = None,
    ) -> dict:
        """Build a detailed human-readable schedule for an experiment."""
        start = start_date or datetime.utcnow()
        schedule = {
            "experiment": experiment.name,
            "start": start.isoformat(),
            "phases": [],
        }

        if experiment.ramp_up_schedule.stages:
            cursor = start
            for stage in experiment.ramp_up_schedule.stages:
                end = cursor + timedelta(hours=stage.duration_hours)
                schedule["phases"].append({
                    "phase": stage.description,
                    "traffic_percent": stage.traffic_percent,
                    "start": cursor.isoformat(),
                    "end": end.isoformat(),
                    "duration_hours": stage.duration_hours,
                })
                cursor = end

            # Steady state phase
            remaining_days = (experiment.estimated_duration_days or 14) - (
                sum(s.duration_hours for s in experiment.ramp_up_schedule.stages) / 24
            )
            if remaining_days > 0:
                end = cursor + timedelta(days=remaining_days)
                schedule["phases"].append({
                    "phase": "Full traffic (steady state)",
                    "traffic_percent": 100.0,
                    "start": cursor.isoformat(),
                    "end": end.isoformat(),
                    "duration_hours": remaining_days * 24,
                })
                schedule["end"] = end.isoformat()
        else:
            end = start + timedelta(days=experiment.estimated_duration_days or 14)
            schedule["phases"].append({
                "phase": "Full traffic",
                "traffic_percent": 100.0,
                "start": start.isoformat(),
                "end": end.isoformat(),
            })
            schedule["end"] = end.isoformat()

        return schedule

    # ── reporting ────────────────────────────────────────────────────

    def get_assignment_counts(self, experiment_id: str) -> dict[str, int]:
        """Return per-variant assignment counts."""
        counts: dict[str, int] = {}
        for a in self._assignments.get(experiment_id, []):
            counts[a.variant_id] = counts.get(a.variant_id, 0) + 1
        return counts

    def get_all_assignments(self, experiment_id: str) -> list[Assignment]:
        return list(self._assignments.get(experiment_id, []))

    # ── persistence ──────────────────────────────────────────────────

    def save_state(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else self._storage_path
        if not p:
            return
        data = {
            eid: [a.model_dump(mode="json") for a in assigns]
            for eid, assigns in self._assignments.items()
        }
        p.write_text(json.dumps(data, indent=2, default=str))

    def load_state(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else self._storage_path
        if not p or not p.exists():
            return
        data = json.loads(p.read_text())
        for eid, assigns in data.items():
            self._assignments[eid] = [Assignment(**a) for a in assigns]

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _hash_to_bucket(experiment_id: str, user_id: str) -> float:
        """Deterministic hash of (experiment, user) -> [0, 1)."""
        raw = f"{experiment_id}:{user_id}".encode()
        h = hashlib.sha256(raw).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def _select_variant(self, experiment: Experiment, bucket: float, traffic_fraction: float):
        """Map a [0, traffic_fraction) bucket value to a variant."""
        # Normalize bucket into [0, 1) range within the traffic window
        normalized = bucket / traffic_fraction if traffic_fraction > 0 else 0

        cumulative = 0.0
        total_pct = experiment.total_traffic_allocated()
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage / total_pct
            if normalized < cumulative:
                return variant
        return experiment.variants[-1] if experiment.variants else None

    def _find_existing_assignment(
        self, experiment_id: str, user_id: str
    ) -> Assignment | None:
        for a in self._assignments.get(experiment_id, []):
            if a.user_id == user_id:
                return a
        return None

    def _current_traffic_fraction(self, experiment: Experiment) -> float:
        """Return fraction [0, 1] of traffic that should be included now."""
        stage = self.get_current_ramp_stage(experiment)
        if stage is None:
            return 1.0
        return stage.traffic_percent / 100.0
