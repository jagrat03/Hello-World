# Automated AB Testing Framework

End-to-end A/B testing agent that designs experiments, manages traffic splits, schedules ramp-ups, and runs full statistical analysis (frequentist + Bayesian).

## What It Does

- **Experiment Design** - Power analysis, sample size calculation, automatic variant configuration
- **Traffic Splitting** - Deterministic hash-based user assignment with gradual ramp-up scheduling
- **Statistical Analysis** - Frequentist (z-test, Welch's t-test) + Bayesian (Monte Carlo posterior sampling)
- **Sequential Testing** - O'Brien-Fleming-style early stopping and futility detection
- **Guardrail Monitoring** - Automatically checks that safety metrics don't degrade
- **AI-Powered** - Optional LLM integration for natural language experiment design (set `OPENAI_API_KEY`)
- **Actionable Recommendations** - Automatic ship/no-ship/keep-running decisions with confidence levels

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
# Design an experiment
ab-agent design checkout_redesign \
  --metric conversion_rate \
  --baseline 0.12 \
  --mde 0.05 \
  --traffic 50000 \
  --variants new_checkout \
  --hypothesis "New checkout flow increases conversion"

# Simulate data and analyze
ab-agent simulate checkout_redesign
ab-agent analyze checkout_redesign

# Run the full end-to-end demo
ab-agent demo

# Interactive chat mode
ab-agent interactive
```

### Python API

```python
from ab_testing_agent.agent import ABTestingAgent
from ab_testing_agent.models import MetricType

agent = ABTestingAgent()

# Design
exp = agent.design_from_params(
    name="pricing_test",
    hypothesis="Higher price tier increases ARPU without hurting conversion",
    primary_metric="revenue_per_user",
    metric_type=MetricType.CONTINUOUS,
    baseline_rate=49.99,
    minimum_detectable_effect=0.03,
    daily_traffic=20000,
    variant_names=["price_55", "price_59"],
    guardrail_metrics=[{"name": "conversion_rate", "max_degradation_percent": 5.0}],
)

# Simulate + Analyze
agent.simulate_data(exp.id)
results = agent.analyze_experiment(exp.id)

print(results.summary)
print(results.recommendation)
```

### Natural Language (with LLM)

```python
agent = ABTestingAgent()  # set OPENAI_API_KEY env var

print(agent.handle_message(
    "Design an A/B test for a new onboarding flow. "
    "Current signup rate is 8%, we want to detect a 10% relative improvement. "
    "We get about 30k visitors per day."
))
```

## Architecture

```
ab_testing_agent/
  models/         # Pydantic data models (Experiment, Variant, Event, Results)
  designer/       # Sample size calc, power analysis, ramp-up scheduling
  splitter/       # Hash-based traffic splitting, experiment lifecycle
  analyzer/       # Frequentist + Bayesian stats, guardrails, early stopping
  agent/          # AI orchestrator tying everything together
  cli.py          # Click CLI with Rich output
```

## Analysis Output

The analyzer produces:

- **Per-variant stats**: sample size, mean, std dev, 95% CI
- **Frequentist tests**: p-values, significance flags, confidence intervals of difference
- **Bayesian analysis**: probability of being best, expected loss, credible intervals
- **Guardrail checks**: pass/fail with degradation percentages
- **Early stopping**: whether you can call the test early (or for futility)
- **Recommendation**: SHIP / KEEP RUNNING / NO SIGNIFICANT DIFFERENCE / DO NOT SHIP

## Tests

```bash
pytest tests/ -v
```

68 tests covering all modules: designer, splitter, analyzer, and end-to-end agent flows.
