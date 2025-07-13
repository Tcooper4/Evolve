import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import questionary
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from typer import Option, Typer

from system.infra.agents.core.models.task import (
    TaskPriority,
    TaskStatus,
    TaskType,
)
from trading.automation_api import AutomationAPI
from trading.automation_core import AutomationCore
from trading.automation_monitoring import AutomationMonitoring
from trading.automation_notification import AutomationNotification
from trading.automation_scheduler import AutomationScheduler
from trading.automation_tasks import AutomationTasks
from trading.automation_workflows import AutomationWorkflows

logger = logging.getLogger(__name__)


class CLIConfig(BaseModel):
    """Configuration for CLI."""

    theme: str = Field(default="dark")
    show_progress: bool = Field(default=True)
    show_timestamps: bool = Field(default=True)
    max_table_rows: int = Field(default=50)
    cache_size: int = Field(default=1000)
    cache_ttl: int = Field(default=3600)


class AutomationCLI:
    """Command-line interface functionality."""

    def __init__(
        self,
        core: AutomationCore,
        tasks: AutomationTasks,
        workflows: AutomationWorkflows,
        monitoring: AutomationMonitoring,
        notification: AutomationNotification,
        scheduler: AutomationScheduler,
        api: AutomationAPI,
        config_path: str = "automation/config/cli.json",
    ):
        """Initialize CLI."""
        self.core = core
        self.tasks = tasks
        self.workflows = workflows
        self.monitoring = monitoring
        self.notification = notification
        self.scheduler = scheduler
        self.api = api
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.setup_cli()
        self.console = Console(theme=self.config.theme)
        self.lock = asyncio.Lock()

    def _load_config(self, config_path: str) -> CLIConfig:
        """Load CLI configuration."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return CLIConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to load CLI config: {str(e)}")
            raise

    def setup_logging(self):
        """Configure logging."""
        log_path = Path("automation/logs")
        log_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_path / "cli.log"), logging.StreamHandler()],
        )

    def setup_cli(self):
        """Setup CLI application."""
        self.app = Typer(name="automation", help="Automation system CLI", add_completion=False)
        # Add commands
        self._add_commands()

    def _add_commands(self):
        """Add CLI commands."""

        @self.app.command()
        def tasks(
            status: Optional[TaskStatus] = Option(None, help="Filter by task status"),
            task_type: Optional[TaskType] = Option(None, help="Filter by task type"),
            limit: int = Option(50, help="Maximum number of tasks to show"),
        ):
            """List tasks."""
            try:
                tasks = asyncio.run(self.tasks.list_tasks(status, task_type))
                table = Table(title="Tasks", show_header=True, header_style="bold magenta")
                table.add_column("ID", style="dim")
                table.add_column("Name")
                table.add_column("Type")
                table.add_column("Status")
                table.add_column("Priority")
                table.add_column("Created")
                for task in tasks[:limit]:
                    table.add_row(
                        task.id,
                        task.name,
                        task.type.value,
                        task.status.value,
                        task.priority.value,
                        task.created_at.isoformat(),
                    )
                self.console.print(table)
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def create_task(
            name: str = Option(..., help="Task name"),
            description: str = Option(..., help="Task description"),
            task_type: TaskType = Option(..., help="Task type"),
            priority: TaskPriority = Option(TaskPriority.NORMAL, help="Task priority"),
        ):
            """Create a new task."""
            try:
                task = asyncio.run(
                    self.tasks.schedule_task(name=name, description=description, task_type=task_type, priority=priority)
                )
                self.console.print(
                    Panel(
                        f"Task created successfully:\n"
                        f"ID: {task.id}\n"
                        f"Name: {task.name}\n"
                        f"Type: {task.type.value}\n"
                        f"Status: {task.status.value}\n"
                        f"Priority: {task.priority.value}",
                        title="Success",
                        border_style="green",
                    )
                )
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def workflows(
            status: Optional[str] = Option(None, help="Filter by workflow status"),
            limit: int = Option(50, help="Maximum number of workflows to show"),
        ):
            """List workflows."""
            try:
                workflows = asyncio.run(self.workflows.get_workflows(status))
                table = Table(title="Workflows", show_header=True, header_style="bold magenta")
                table.add_column("ID", style="dim")
                table.add_column("Name")
                table.add_column("Status")
                table.add_column("Steps")
                table.add_column("Created")
                for workflow in workflows[:limit]:
                    table.add_row(
                        workflow.id,
                        workflow.name,
                        workflow.status,
                        str(len(workflow.steps)),
                        workflow.created_at.isoformat(),
                    )
                self.console.print(table)
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def create_workflow(
            name: str = Option(..., help="Workflow name"), description: str = Option(..., help="Workflow description")
        ):
            """Create a new workflow."""
            try:
                # Get steps interactively
                steps = []
                while True:
                    step_name = questionary.text("Enter step name (or press Enter to finish):").ask()

                    if not step_name:
                        break

                    task_type = questionary.select("Select task type:", choices=[t.value for t in TaskType]).ask()

                    steps.append({"name": step_name, "task_type": TaskType(task_type)})

                workflow = asyncio.run(self.workflows.create_workflow(name=name, description=description, steps=steps))

                self.console.print(
                    Panel(
                        f"Workflow created successfully:\n"
                        f"ID: {workflow.id}\n"
                        f"Name: {workflow.name}\n"
                        f"Steps: {len(workflow.steps)}",
                        title="Success",
                        border_style="green",
                    )
                )

            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def metrics():
            """Show system metrics."""
            try:
                metrics = asyncio.run(self.monitoring.get_metrics())

                self.console.print(
                    Panel(
                        Markdown(
                            f"# System Metrics\n\n"
                            f"## Tasks\n"
                            f"- Total: {metrics['tasks']['total']}\n"
                            f"- Queue Size: {metrics['tasks']['queue_size']}\n\n"
                            f"## Workflows\n"
                            f"- Total: {metrics['workflows']['total']}\n"
                            f"- Queue Size: {metrics['workflows']['queue_size']}\n\n"
                            f"## System\n"
                            f"- CPU Usage: {metrics['system']['cpu_usage']}%\n"
                            f"- Memory Usage: {metrics['system']['memory_usage']}%\n"
                            f"- Disk Usage: {metrics['system']['disk_usage']}%\n\n"
                            f"## Errors\n"
                            f"- Total: {metrics['errors']}"
                        ),
                        title="Metrics",
                        border_style="blue",
                    )
                )

            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def schedules(
            enabled: Optional[bool] = Option(None, help="Filter by enabled status"),
            task_type: Optional[TaskType] = Option(None, help="Filter by task type"),
            limit: int = Option(50, help="Maximum number of schedules to show"),
        ):
            """List schedules."""
            try:
                schedules = asyncio.run(self.scheduler.get_schedules(enabled, task_type))

                table = Table(title="Schedules", show_header=True, header_style="bold magenta")

                table.add_column("ID", style="dim")
                table.add_column("Name")
                table.add_column("Type")
                table.add_column("Enabled")
                table.add_column("Next Run")
                table.add_column("Created")

                for schedule in schedules[:limit]:
                    table.add_row(
                        schedule.id,
                        schedule.name,
                        schedule.task_type.value,
                        str(schedule.enabled),
                        schedule.next_run.isoformat() if schedule.next_run else "N/A",
                        schedule.created_at.isoformat(),
                    )

                self.console.print(table)

            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def create_schedule(
            name: str = Option(..., help="Schedule name"),
            description: str = Option(..., help="Schedule description"),
            task_type: TaskType = Option(..., help="Task type"),
            cron_expression: Optional[str] = Option(None, help="Cron expression"),
            interval_seconds: Optional[int] = Option(None, help="Interval in seconds"),
        ):
            """Create a new schedule."""
            try:
                schedule = asyncio.run(
                    self.scheduler.create_schedule(
                        name=name,
                        description=description,
                        task_type=task_type,
                        parameters={},
                        cron_expression=cron_expression,
                        interval_seconds=interval_seconds,
                    )
                )

                self.console.print(
                    Panel(
                        f"Schedule created successfully:\n"
                        f"ID: {schedule.id}\n"
                        f"Name: {schedule.name}\n"
                        f"Type: {schedule.task_type.value}\n"
                        f"Next Run: {schedule.next_run.isoformat() if schedule.next_run else 'N/A'}",
                        title="Success",
                        border_style="green",
                    )
                )

            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def start_api():
            """Start the API server."""
            try:
                asyncio.run(self.api.start())
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

    def run(self):
        """Run the CLI application."""
        try:
            self.app()
        except Exception as e:
            logger.error(f"CLI error: {str(e)}")
            self.console.print(f"[red]Error: {str(e)}[/red]")

    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear console
            self.console.clear()

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
