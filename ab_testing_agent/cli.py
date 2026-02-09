"""CLI interface for the A/B Testing AI Agent."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ab_testing_agent.agent.orchestrator import ABTestingAgent

console = Console()


@click.group()
@click.option("--api-key", envvar="OPENAI_API_KEY", default=None, help="OpenAI API key for LLM features")
@click.pass_context
def main(ctx: click.Context, api_key: str | None) -> None:
    """AB Testing AI Agent - Design, split, and analyze A/B tests end-to-end."""
    ctx.ensure_object(dict)
    ctx.obj["agent"] = ABTestingAgent(openai_api_key=api_key)


@main.command()
@click.argument("name")
@click.option("--metric", "-m", default="conversion_rate", help="Primary metric name")
@click.option("--type", "metric_type", default="conversion", type=click.Choice(["conversion", "continuous", "count", "ratio"]))
@click.option("--baseline", "-b", type=float, default=0.10, help="Baseline metric value")
@click.option("--mde", type=float, default=0.05, help="Minimum detectable effect (relative)")
@click.option("--traffic", "-t", type=int, default=10000, help="Daily traffic estimate")
@click.option("--variants", "-v", default=None, help="Comma-separated variant names")
@click.option("--alpha", type=float, default=0.05, help="Significance level")
@click.option("--power", type=float, default=0.80, help="Statistical power")
@click.option("--hypothesis", "-h", default="", help="Experiment hypothesis")
@click.pass_context
def design(
    ctx: click.Context,
    name: str,
    metric: str,
    metric_type: str,
    baseline: float,
    mde: float,
    traffic: int,
    variants: str | None,
    alpha: float,
    power: float,
    hypothesis: str,
) -> None:
    """Design a new A/B test experiment."""
    agent: ABTestingAgent = ctx.obj["agent"]

    variant_list = [v.strip() for v in variants.split(",")] if variants else None
    cmd = (
        f"design {name} metric={metric} type={metric_type} "
        f"baseline={baseline} mde={mde} traffic={traffic} "
        f"alpha={alpha} power={power}"
    )
    if hypothesis:
        cmd += f" hypothesis={hypothesis}"

    # Use the agent's design_from_params for more control
    from ab_testing_agent.models.experiment import MetricType as MT
    exp = agent.design_from_params(
        name=name,
        hypothesis=hypothesis or "Treatment improves the primary metric.",
        primary_metric=metric,
        metric_type=MT(metric_type),
        baseline_rate=baseline,
        minimum_detectable_effect=mde,
        daily_traffic=traffic,
        variant_names=variant_list,
        alpha=alpha,
        power=power,
    )

    _print_experiment(exp)


@main.command()
@click.argument("experiment_name")
@click.pass_context
def start(ctx: click.Context, experiment_name: str) -> None:
    """Start an experiment (begin traffic splitting)."""
    agent: ABTestingAgent = ctx.obj["agent"]
    result = agent.handle_message(f"start {experiment_name}")
    console.print(Panel(result, title="Experiment Started", border_style="green"))


@main.command()
@click.argument("experiment_name")
@click.pass_context
def stop(ctx: click.Context, experiment_name: str) -> None:
    """Stop an experiment early."""
    agent: ABTestingAgent = ctx.obj["agent"]
    result = agent.handle_message(f"stop {experiment_name}")
    console.print(Panel(result, title="Experiment Stopped", border_style="red"))


@main.command()
@click.argument("experiment_name")
@click.pass_context
def schedule(ctx: click.Context, experiment_name: str) -> None:
    """View the experiment schedule and ramp-up plan."""
    agent: ABTestingAgent = ctx.obj["agent"]
    result = agent.handle_message(f"schedule {experiment_name}")
    console.print(Panel(result, title="Schedule", border_style="blue"))


@main.command()
@click.argument("experiment_name")
@click.option("--samples", "-n", type=int, default=None, help="Samples per variant")
@click.pass_context
def simulate(ctx: click.Context, experiment_name: str, samples: int | None) -> None:
    """Generate simulated data for an experiment."""
    agent: ABTestingAgent = ctx.obj["agent"]
    if samples:
        exp = None
        for e in agent.list_experiments():
            if experiment_name.lower() in e.name.lower():
                exp = e
                break
        if exp:
            result = agent.simulate_data(exp.id, n_per_variant=samples)
        else:
            result = f"Experiment '{experiment_name}' not found."
    else:
        result = agent.handle_message(f"simulate {experiment_name}")
    console.print(Panel(result, title="Simulation", border_style="yellow"))


@main.command()
@click.argument("experiment_name")
@click.pass_context
def analyze(ctx: click.Context, experiment_name: str) -> None:
    """Run full statistical analysis on an experiment."""
    agent: ABTestingAgent = ctx.obj["agent"]
    result = agent.handle_message(f"analyze {experiment_name}")
    console.print(Panel(result, title="Analysis Results", border_style="cyan"))


@main.command(name="status")
@click.pass_context
def status_cmd(ctx: click.Context) -> None:
    """View status of all experiments."""
    agent: ABTestingAgent = ctx.obj["agent"]
    result = agent.handle_message("status")
    console.print(Panel(result, title="Status", border_style="white"))


@main.command(name="list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List all experiments."""
    agent: ABTestingAgent = ctx.obj["agent"]
    result = agent.handle_message("list")
    console.print(result)


@main.command()
@click.pass_context
def interactive(ctx: click.Context) -> None:
    """Start an interactive chat session with the agent."""
    agent: ABTestingAgent = ctx.obj["agent"]
    console.print(
        Panel(
            "AB Testing AI Agent - Interactive Mode\n"
            "Type 'help' for commands, 'quit' to exit.",
            title="Welcome",
            border_style="bold green",
        )
    )

    while True:
        try:
            user_input = console.input("[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("Goodbye!")
            break

        response = agent.handle_message(user_input)
        console.print(Panel(response, border_style="dim"))


@main.command()
def demo() -> None:
    """Run a full end-to-end demo."""
    agent = ABTestingAgent()

    console.print(Panel("A/B Testing Agent - End-to-End Demo", style="bold green"))
    console.print()

    # Step 1: Design
    console.print("[bold]Step 1: Design Experiment[/]")
    result = agent.handle_message(
        "design checkout_redesign metric=conversion_rate baseline=0.12 mde=0.05 traffic=50000 "
        "variants=new_checkout hypothesis=New checkout flow increases conversion"
    )
    console.print(Panel(result, title="Design", border_style="blue"))
    console.print()

    # Step 2: Schedule
    console.print("[bold]Step 2: View Schedule[/]")
    result = agent.handle_message("schedule checkout_redesign")
    console.print(Panel(result, title="Schedule", border_style="yellow"))
    console.print()

    # Step 3: Start
    console.print("[bold]Step 3: Start Experiment[/]")
    result = agent.handle_message("start checkout_redesign")
    console.print(Panel(result, title="Started", border_style="green"))
    console.print()

    # Step 4: Simulate data
    console.print("[bold]Step 4: Simulate Data[/]")
    result = agent.handle_message("simulate checkout_redesign")
    console.print(Panel(result, title="Simulation", border_style="yellow"))
    console.print()

    # Step 5: Analyze
    console.print("[bold]Step 5: Full Analysis[/]")
    result = agent.handle_message("analyze checkout_redesign")
    console.print(Panel(result, title="Analysis", border_style="cyan"))
    console.print()

    console.print("[bold green]Demo complete![/]")


def _print_experiment(exp) -> None:
    """Pretty-print an experiment configuration."""
    table = Table(title=f"Experiment: {exp.name}", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("ID", exp.id)
    table.add_row("Status", exp.status.value)
    table.add_row("Hypothesis", exp.hypothesis)
    table.add_row("Primary Metric", f"{exp.primary_metric} ({exp.primary_metric_type.value})")
    table.add_row("MDE", f"{exp.minimum_detectable_effect:.2%}")
    table.add_row("Significance Level", f"{exp.significance_level}")
    table.add_row("Power", f"{exp.statistical_power}")
    table.add_row("Required Sample/Variant", f"{exp.required_sample_size_per_variant:,}")
    table.add_row("Estimated Duration", f"{exp.estimated_duration_days:.0f} days")
    table.add_row("Daily Traffic", f"{exp.daily_traffic_estimate:,}")

    variants_str = "\n".join(
        f"  {v.name} ({'control' if v.is_control else 'treatment'}) - {v.traffic_percentage:.1f}%"
        for v in exp.variants
    )
    table.add_row("Variants", variants_str)

    if exp.ramp_up_schedule.stages:
        ramp = "\n".join(
            f"  {s.description} ({s.duration_hours:.0f}h)"
            for s in exp.ramp_up_schedule.stages
        )
        table.add_row("Ramp-up", ramp)

    console.print(table)


if __name__ == "__main__":
    main()
