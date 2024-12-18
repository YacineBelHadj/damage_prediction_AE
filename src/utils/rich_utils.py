from rich.console import Console
from rich.table import Table

def print_stats_as_table(statistics: dict):
    console = Console()
    table = Table(title="Statistics Table")

    # Define table columns
    table.add_column("Column Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Statistic", justify="left", style="red")
    table.add_column("Value", justify="right", style="green")

    # Add rows to the table
    for column, stats in statistics.items():
        for stat_name, value in stats.items():
            table.add_row(column, stat_name, f"{value:.4f}" if isinstance(value, (int, float)) else str(value))

    console.print(table)