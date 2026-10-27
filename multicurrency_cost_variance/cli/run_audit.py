"""
Rich Terminal Audit Dashboard.

Orchestrates the full pipeline — ECB download → ledger generation → ETL →
Monte Carlo simulation — and displays results in a professional Rich
terminal dashboard.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on the path when running as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box

from etl.fetch_ecb_fx import load_fx_rates
from etl.ledger_generator import generate_ledger
from etl.etl_pipeline import run_etl, get_etl_summary
from simulation.cost_drivers import get_all_drivers
from simulation.monte_carlo import (
    run_monte_carlo,
    compute_var,
    sensitivity_analysis,
    tornado_data,
)

console = Console()


def _fmt_eur(value: float) -> str:
    """Format a number as EUR with thousand separators."""
    if abs(value) >= 1_000_000_000:
        return f"€{value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"€{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"€{value / 1_000:,.1f}K"
    return f"€{value:,.2f}"


def _pct(value: float) -> str:
    """Format a percentage."""
    return f"{value:,.2f}%"


def build_etl_panel(stats: dict, enriched_df, dlq_df) -> Panel:
    """Build the ETL Ingestion Report panel."""
    table = Table(
        title="",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Metric", style="bold white", min_width=28)
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Records Processed", f"{stats['total_records']:,}")
    table.add_row("Matched Records", f"{stats['matched_records']:,}")
    table.add_row("DLQ (Rejected) Records", f"{stats['dlq_records']:,}")
    table.add_row("Reconciliation Rate", _pct(stats["reconciliation_rate"]))
    table.add_row("Currencies Processed", str(len(stats["currencies_processed"])))
    table.add_row("Currency List", ", ".join(stats["currencies_processed"][:10]))
    table.add_row("Execution Time", f"{stats['execution_time_seconds']:.3f}s")
    table.add_row("Total EUR Exposure", _fmt_eur(stats["total_eur_exposure"]))
    table.add_row("Total USD Exposure", _fmt_eur(stats["total_usd_exposure"]))

    return Panel(
        table,
        title="[bold blue]📊 ETL Ingestion Report[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )


def build_risk_panel(var_results: dict) -> Panel:
    """Build the Financial Risk Summary panel."""
    table = Table(
        title="",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Metric", style="bold white", min_width=28)
    table.add_column("Value", justify="right", style="yellow")

    base = var_results["base_budget"]
    expected = var_results["expected_budget"]
    variance = expected - base
    variance_pct = (variance / base) * 100 if base != 0 else 0

    table.add_row("Base Budget (Median)", _fmt_eur(base))
    table.add_row("Expected Budget (Mean)", _fmt_eur(expected))
    table.add_row("Budget Variance", f"{_fmt_eur(variance)}  ({variance_pct:+.2f}%)")
    table.add_row("Std Deviation", _fmt_eur(var_results["std_dev"]))
    table.add_row("VaR 95%", _fmt_eur(var_results["var_95"]))
    table.add_row("VaR 99%", _fmt_eur(var_results["var_99"]))
    table.add_row("Max Simulated Cost", _fmt_eur(var_results["max_cost"]))
    table.add_row("Min Simulated Cost", _fmt_eur(var_results["min_cost"]))

    return Panel(
        table,
        title="[bold magenta]⚠️  Financial Risk Summary (Monte Carlo 10K)[/bold magenta]",
        border_style="magenta",
        padding=(1, 2),
    )


def build_sensitivity_panel(sens_df) -> Panel:
    """Build the Cost Driver Sensitivity panel with visual bars."""
    table = Table(
        title="",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold red",
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Cost Driver", style="bold white", min_width=22)
    table.add_column("Category", style="cyan", min_width=10)
    table.add_column("Spearman ρ", justify="right", style="yellow", min_width=10)
    table.add_column("Impact", min_width=30)

    top5 = sens_df.head(5)
    max_corr = top5["abs_corr"].max() if len(top5) > 0 else 1.0

    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        bar_width = int((row["abs_corr"] / max_corr) * 25) if max_corr > 0 else 0
        bar = Text("█" * bar_width + "░" * (25 - bar_width), style="red")
        table.add_row(
            str(rank),
            row["driver"],
            row["category"],
            f"{row['spearman_corr']:.4f}",
            bar,
        )

    return Panel(
        table,
        title="[bold red]🔥 Cost Driver Sensitivity (Top 5)[/bold red]",
        border_style="red",
        padding=(1, 2),
    )


def build_currency_panel(enriched_df) -> Panel:
    """Build the Currency Exposure Summary panel."""
    table = Table(
        title="",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold green",
        expand=True,
    )
    table.add_column("Currency", style="bold white", min_width=10)
    table.add_column("Transactions", justify="right", style="cyan", min_width=14)
    table.add_column("Total EUR Exposure", justify="right", style="yellow", min_width=18)
    table.add_column("% of Total", justify="right", style="green", min_width=10)

    currency_stats = (
        enriched_df.groupby("currency")
        .agg(count=("transaction_id", "count"), total_eur=("amount_eur", "sum"))
        .sort_values("total_eur", ascending=False)
    )
    grand_total = currency_stats["total_eur"].sum()

    for currency, row in currency_stats.head(10).iterrows():
        pct = (row["total_eur"] / grand_total * 100) if grand_total > 0 else 0
        table.add_row(
            str(currency),
            f"{int(row['count']):,}",
            _fmt_eur(row["total_eur"]),
            _pct(pct),
        )

    return Panel(
        table,
        title="[bold green]💱 Currency Exposure Summary[/bold green]",
        border_style="green",
        padding=(1, 2),
    )


def main() -> None:
    """Run the full audit pipeline and display the Rich dashboard."""
    data_dir = str(_project_root / "data")

    console.print()
    console.rule("[bold blue]Multi-Currency Transaction ETL & Parametric Cost Variance Engine[/bold blue]")
    console.print(
        "[dim]Author: Eyoas Gizachew Zewdie · University of Messina · B.Sc. Data Analysis[/dim]",
        justify="center",
    )
    console.print()

    # ------------------------------------------------------------------
    # Step 1: Download ECB FX rates
    # ------------------------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Downloading ECB reference rates…", total=1)
        fx_rates = load_fx_rates(data_dir=data_dir)
        progress.update(task, completed=1)

    console.print(f"  ✅  FX rates loaded: [green]{len(fx_rates):,}[/green] rows, "
                  f"[green]{fx_rates['currency_code'].nunique()}[/green] currencies")

    # ------------------------------------------------------------------
    # Step 2: Generate transaction ledger
    # ------------------------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Generating 100K transaction ledger…", total=1)
        ledger = generate_ledger(n_transactions=100_000, output_dir=data_dir)
        progress.update(task, completed=1)

    console.print(f"  ✅  Ledger generated: [green]{len(ledger):,}[/green] transactions")

    # ------------------------------------------------------------------
    # Step 3: Run ETL pipeline
    # ------------------------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running DuckDB ETL pipeline…", total=1)
        enriched, dlq, stats = run_etl(ledger, fx_rates)
        progress.update(task, completed=1)

    console.print(f"  ✅  ETL complete: reconciliation [green]{stats['reconciliation_rate']:.2f}%[/green]")

    # ------------------------------------------------------------------
    # Step 4: Run Monte Carlo simulation
    # ------------------------------------------------------------------
    drivers = get_all_drivers()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running 10,000-iteration Monte Carlo…", total=1)
        simulations = run_monte_carlo(drivers, n_iterations=10_000, seed=42)
        var_results = compute_var(simulations)
        sens = sensitivity_analysis(simulations, drivers)
        tornado = tornado_data(simulations, drivers)
        progress.update(task, completed=1)

    console.print(f"  ✅  Simulation complete: [green]10,000[/green] iterations × "
                  f"[green]15[/green] drivers")
    console.print()

    # ------------------------------------------------------------------
    # Display Dashboard Panels
    # ------------------------------------------------------------------
    console.print(build_etl_panel(stats, enriched, dlq))
    console.print()
    console.print(build_risk_panel(var_results))
    console.print()
    console.print(build_sensitivity_panel(sens))
    console.print()
    console.print(build_currency_panel(enriched))
    console.print()
    console.rule("[bold blue]Audit Complete[/bold blue]")
    console.print()


if __name__ == "__main__":
    main()
