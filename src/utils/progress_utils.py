import time
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

def create_progress_bar() -> Progress:
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )

def print_metrics_table(metrics: dict, epoch: int):
    table = Table(title=f"Epoch {epoch} Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.2f}")
        else:
            table.add_row(key, str(value))

    console.print(table)