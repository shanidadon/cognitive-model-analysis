"""Command-line interface for running ML experiments.

This module provides a comprehensive CLI for the research project template,
allowing users to run experiments, manage configurations, and view results
through a clean command-line interface.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import experiment components
from src.experiment import ExperimentConfig
from src.experiment.pipeline import ExperimentRidge
from src.experiment.schemas import load_config

# Initialize Typer app and Rich console
app = typer.Typer(
    name="experiment-runner",
    help="ML Experiment Runner with Pydantic + ExCa",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(
        ..., help="Path to experiment configuration YAML file"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Validate config and show summary without running",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force run even if cache exists"
    ),
    ridge: bool = typer.Option(
        False,
        "--ridge",
        "-r",
        help="Run ExperimentRidge (DINOv2 + Ridge) instead of standard Experiment",
    ),
) -> None:
    """Run an experiment from a configuration file.

    Examples:
        experiment-runner run configs/local.yaml
        experiment-runner run configs/local.yaml --ridge
        experiment-runner run configs/production.yaml --verbose
        experiment-runner run configs/local.yaml --dry-run
    """
    try:
        # Load and validate configuration
        with console.status("[bold blue]Loading configuration...", spinner="dots"):
            exp_config = load_config(config)

        console.print(f"[green]✓[/green] Configuration loaded: {config}")
        console.print(f"[blue]Experiment:[/blue] {exp_config.experiment.name}")
        console.print(f"[blue]Model:[/blue] {exp_config.model.name}")
        console.print(f"[blue]Device:[/blue] {exp_config.experiment.device}")
        if ridge:
            console.print(f"[blue]Mode:[/blue] Ridge Regression (DINOv2)")

        if dry_run:
            _show_config_summary(exp_config)
            console.print(
                "[yellow]Dry run completed. Use --no-dry-run to actually run the experiment.[/yellow]"
            )
            return

        # Instantiate the appropriate experiment type

        experiment = ExperimentRidge(config=exp_config)

        # Clear cache if forced
        if force:
            if hasattr(experiment, "infra"):
                cleared = experiment.infra.clean_cache(keep_recent=0)
                console.print(f"[yellow]Cleared {cleared} cache files[/yellow]")

        # Run experiment
        console.print("\n[bold]Starting experiment...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running experiment...", total=None)

            if ridge:
                experiment.train()
                results = experiment.evaluate()
            else:
                results = experiment.run()

        # Display results
        _show_results(results)

        console.print(f"\n[green]✓[/green] Experiment completed successfully!")
        console.print(f"[blue]Results saved to:[/blue] {results['results_path']}")

    except Exception as e:
        console.print(f"[red]✗[/red] Experiment failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to configuration file to validate")
) -> None:
    """Validate a configuration file without running the experiment."""
    try:
        with console.status("[bold blue]Validating configuration...", spinner="dots"):
            exp_config = load_config(config)

        console.print(f"[green]✓[/green] Configuration is valid: {config}")
        _show_config_summary(exp_config)

    except Exception as e:
        console.print(f"[red]✗[/red] Configuration validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def list_configs(
    configs_dir: Path = typer.Option(
        Path("configs"), "--dir", "-d", help="Directory to search for configs"
    )
) -> None:
    """List available configuration files."""
    if not configs_dir.exists():
        console.print(f"[red]✗[/red] Configuration directory not found: {configs_dir}")
        raise typer.Exit(1)

    yaml_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

    if not yaml_files:
        console.print(f"[yellow]No configuration files found in {configs_dir}[/yellow]")
        return

    table = Table(title="Available Configurations")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Model", style="green")
    table.add_column("Description", style="white")
    table.add_column("Valid", style="bold")

    for yaml_file in sorted(yaml_files):
        try:
            config = load_config(yaml_file)
            model_name = config.model.name
            description = config.experiment.description or "No description"
            valid = "[green]✓[/green]"
        except Exception:
            model_name = "Unknown"
            description = "Invalid configuration"
            valid = "[red]✗[/red]"

        table.add_row(
            str(yaml_file.name),
            model_name,
            description[:50] + "..." if len(description) > 50 else description,
            valid,
        )

    console.print(table)


@app.command()
def results(
    experiment_name: Optional[str] = typer.Argument(
        None, help="Experiment name to show results for"
    ),
    cache_dir: Path = typer.Option(
        Path("cache"), "--cache-dir", "-c", help="Cache directory to search"
    ),
    list_all: bool = typer.Option(
        False, "--list", "-l", help="List all available experiments"
    ),
) -> None:
    """Show results from completed experiments."""
    if not cache_dir.exists():
        console.print(f"[red]✗[/red] Cache directory not found: {cache_dir}")
        raise typer.Exit(1)

    # Find result files
    result_files = list(cache_dir.glob("**/experiment_results.pkl"))
    run_files = list(cache_dir.glob("**/run_*.json"))

    if list_all or experiment_name is None:
        _list_all_experiments(result_files, run_files)
        return

    # Show specific experiment results
    _show_experiment_results(experiment_name, result_files, run_files)


@app.command()
def clean(
    cache_dir: Path = typer.Option(
        Path("cache"), "--cache-dir", "-c", help="Cache directory to clean"
    ),
    keep: int = typer.Option(5, "--keep", "-k", help="Number of recent files to keep"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force clean without confirmation"
    ),
) -> None:
    """Clean experiment cache files."""
    if not cache_dir.exists():
        console.print(f"[yellow]Cache directory does not exist: {cache_dir}[/yellow]")
        return

    if not force:
        confirm = typer.confirm(
            f"This will delete old cache files in {cache_dir}. Continue?"
        )
        if not confirm:
            console.print("[yellow]Cache cleaning cancelled[/yellow]")
            return

    # Import and create infra for cleaning
    from src.experiment.infra import create_infra

    infra = create_infra(cache_dir, "cleanup")

    with console.status("[bold blue]Cleaning cache...", spinner="dots"):
        deleted_count = infra.clean_cache(keep_recent=keep)

    console.print(f"[green]✓[/green] Deleted {deleted_count} old cache files")
    console.print(f"[blue]Kept {keep} most recent files of each type[/blue]")


@app.command()
def info() -> None:
    """Show information about the experiment framework."""
    info_panel = Panel.fit(
        """[bold]ML Experiment Framework[/bold]

[blue]Features:[/blue]
• Pydantic configuration validation
• ExCa experiment caching and reproducibility  
• CIFAR-10 classification with multiple model types
• Ridge regression with DINOv2 features
• Automated testing with GitHub Actions
• Clean project structure following ML best practices

[blue]Available Models:[/blue]
• Linear: Simple logistic regression
• MLP: Multi-layer perceptron with configurable architecture
• CNN: Convolutional neural network with batch normalization
• DINOv2: Vision foundation model with ridge classifier

[blue]Usage:[/blue]
Run 'experiment-runner --help' for available commands
Run with --ridge flag for DINOv2 + Ridge Regression
Check configs/ directory for example configurations
        """,
        title="🔬 Research Project Template",
        border_style="blue",
    )

    console.print(info_panel)


def _show_config_summary(config: ExperimentConfig) -> None:
    """Show a summary of the experiment configuration."""

    # Create summary table
    table = Table(title="Experiment Configuration Summary")
    table.add_column("Component", style="cyan", width=15)
    table.add_column("Configuration", style="white")

    # Experiment info
    table.add_row(
        "Experiment",
        f"Name: {config.experiment.name}\nSeed: {config.experiment.random_seed}",
    )

    # Data configuration
    data_info = f"Dataset: {config.data.dataset}\nBatch size: {config.data.batch_size}"
    table.add_row("Data", data_info)

    # Model configuration - only DINOv2_ViT_Large is supported
    model_info = f"Type: {config.model.name}"
    table.add_row("Model", model_info)

    # Training configuration (if applicable)
    if hasattr(config, "training") and config.training:
        training_info = f"Epochs: {config.training.epochs}\nLR: {config.training.learning_rate}\nOptimizer: {config.training.optimizer}"
        if config.training.scheduler:
            training_info += f"\nScheduler: {config.training.scheduler.name}"
        table.add_row("Training", training_info)

    console.print(table)


def _show_results(results: dict) -> None:
    """Display experiment results in a formatted way."""
    console.print("\n[bold]Experiment Results[/bold]")

    # Create results table
    table = Table()
    table.add_column("Split", style="cyan", width=12)
    table.add_column("Metric", style="green")
    table.add_column("Value", style="white", justify="right")

    for key, value in results.items():
        if key.endswith("_metrics") and isinstance(value, dict):
            split_name = key.replace("_metrics", "").title()
            for metric_name, metric_value in value.items():
                if isinstance(metric_value, (int, float)):
                    if metric_name == "loss":
                        formatted_value = f"{metric_value:.4f}"
                    else:
                        formatted_value = f"{metric_value:.3f}"
                    table.add_row(split_name, metric_name.title(), formatted_value)
                    split_name = ""  # Only show split name for first metric

    console.print(table)

    # Show timing info
    if "experiment_time" in results:
        time_mins = results["experiment_time"] / 60
        console.print(f"\n[blue]Experiment time:[/blue] {time_mins:.2f} minutes")

    # Save results to CSV
    _save_results_to_csv(results)

    # Generate and display graphs
    _plot_training_graphs(results)


def _save_results_to_csv(results: dict) -> None:
    """Save model name and accuracy to CSV file."""
    try:
        import csv
    except ImportError:
        console.print(
            "[yellow]Warning: csv module not available. Skipping CSV save.[/yellow]"
        )
        return

    # Extract model name from results
    model_name = results.get("model_name", "Unknown")

    # Extract accuracy metrics
    test_accuracy = None
    train_accuracy = None

    if "test_metrics" in results and isinstance(results["test_metrics"], dict):
        test_accuracy = results["test_metrics"].get("test_accuracy")

    if "train_metrics" in results and isinstance(results["train_metrics"], dict):
        train_accuracy = results["train_metrics"].get("train_accuracy")

    # Create results directory if it doesn't exist
    results_dir = Path(results.get("results_path", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "model_accuracy.csv"

    # Check if CSV file exists to decide on header
    file_exists = csv_path.exists()

    try:
        with open(csv_path, "a", newline="") as csvfile:
            fieldnames = ["Model Name", "Train Accuracy", "Test Accuracy", "Timestamp"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Write data row
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow(
                {
                    "Model Name": model_name,
                    "Train Accuracy": (
                        f"{train_accuracy:.4f}" if train_accuracy else "N/A"
                    ),
                    "Test Accuracy": f"{test_accuracy:.4f}" if test_accuracy else "N/A",
                    "Timestamp": timestamp,
                }
            )

        console.print(f"\n[green]✓[/green] Results saved to CSV: {csv_path}")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save CSV file: {e}[/yellow]")


def _plot_training_graphs(results: dict) -> None:
    """Generate performance visualization graphs.

    Supports both epoch-based training history and ridge regression metrics.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print(
            "[yellow]Warning: matplotlib not installed. Skipping graphs.[/yellow]"
        )
        return

    results_dir = Path(results.get("results_path", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # For epoch-based training (standard models)
    if "train_history" in results and results["train_history"]:
        history = results["train_history"]
        epochs = history.get("epochs", [])

        if epochs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            if history.get("train_loss"):
                ax1.plot(
                    epochs,
                    history["train_loss"],
                    marker="o",
                    label="Train Loss",
                    linewidth=2,
                )
            if history.get("val_loss"):
                ax1.plot(
                    epochs,
                    history["val_loss"],
                    marker="s",
                    label="Validation Loss",
                    linewidth=2,
                )
            ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
            ax1.set_title("Loss vs Epochs", fontsize=14, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            if history.get("train_accuracy"):
                ax2.plot(
                    epochs,
                    history["train_accuracy"],
                    marker="o",
                    label="Train Accuracy",
                    linewidth=2,
                )
            if history.get("val_accuracy"):
                ax2.plot(
                    epochs,
                    history["val_accuracy"],
                    marker="s",
                    label="Validation Accuracy",
                    linewidth=2,
                )
            ax2.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
            ax2.set_title("Accuracy vs Epochs", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            graph_path = results_dir / "training_curves.png"
            plt.savefig(graph_path, dpi=150, bbox_inches="tight")
            console.print(f"\n[green]✓[/green] Training curves saved to: {graph_path}")
            plt.close()
            return

    # For ridge regression (non-iterative models)
    if "test_metrics" in results or "train_metrics" in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_data = {}
        if "train_metrics" in results:
            for key, val in results["train_metrics"].items():
                if isinstance(val, (int, float)):
                    metrics_data[f"Train {key.title()}"] = val

        if "test_metrics" in results:
            for key, val in results["test_metrics"].items():
                if isinstance(val, (int, float)):
                    metrics_data[f"Test {key.title()}"] = val

        if metrics_data:
            names = list(metrics_data.keys())
            values = list(metrics_data.values())
            colors = ["#2ecc71" if "train" in n.lower() else "#3498db" for n in names]

            bars = ax.bar(
                names, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
            )
            ax.set_ylabel("Score", fontsize=12, fontweight="bold")
            ax.set_title("Model Performance Metrics", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis="y")

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            graph_path = results_dir / "metrics_summary.png"
            plt.savefig(graph_path, dpi=150, bbox_inches="tight")
            console.print(f"\n[green]✓[/green] Metrics summary saved to: {graph_path}")
            plt.close()


def _list_all_experiments(result_files: List[Path], run_files: List[Path]) -> None:
    """List all available experiments."""
    if not result_files and not run_files:
        console.print("[yellow]No experiment results found[/yellow]")
        return

    table = Table(title="Completed Experiments")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Test Accuracy", style="white", justify="right")
    table.add_column("Date", style="blue")

    # Process run files for basic info
    for run_file in sorted(run_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(run_file) as f:
                run_data = json.load(f)

            exp_name = run_data.get("experiment_name", "Unknown")
            model_name = (
                run_data.get("config", {}).get("model", {}).get("name", "Unknown")
            )
            test_acc = (
                run_data.get("results", {})
                .get("test_metrics", {})
                .get("accuracy", "N/A")
            )
            timestamp = run_data.get("timestamp", 0)

            if isinstance(test_acc, float):
                test_acc = f"{test_acc:.3f}"

            import datetime

            date_str = datetime.datetime.fromtimestamp(timestamp).strftime(
                "%Y-%m-%d %H:%M"
            )

            table.add_row(exp_name, model_name, str(test_acc), date_str)

        except Exception:
            continue  # Skip corrupted files

    console.print(table)


def _show_experiment_results(
    name: str, result_files: List[Path], run_files: List[Path]
) -> None:
    """Show detailed results for a specific experiment."""
    # Search for matching experiment
    matching_runs = []
    for run_file in run_files:
        try:
            with open(run_file) as f:
                run_data = json.load(f)
            if run_data.get("experiment_name") == name:
                matching_runs.append((run_file, run_data))
        except Exception:
            continue

    if not matching_runs:
        console.print(f"[red]✗[/red] No results found for experiment: {name}")
        return

    # Show most recent run
    latest_run = max(matching_runs, key=lambda x: x[1].get("timestamp", 0))
    run_file, run_data = latest_run

    console.print(f"[bold]Results for: {name}[/bold]")

    # Show configuration summary if available
    if "config" in run_data:
        config_dict = run_data["config"]
        console.print(
            f"[blue]Model:[/blue] {config_dict.get('model', {}).get('name', 'Unknown')}"
        )
        console.print(
            f"[blue]Epochs:[/blue] {config_dict.get('training', {}).get('epochs', 'Unknown')}"
        )

    # Show results
    if "results" in run_data:
        _show_results(run_data["results"])


def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
