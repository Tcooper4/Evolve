"""
CLI Service

Implements a command-line interface for managing tasks, workflows, metrics, and schedules in the agentic system.
Adapted from legacy automation/services/automation_cli.py.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
import argparse
import sys
import os
import shutil
import subprocess
from datetime import datetime
import getpass
import readline
import cmd
import colorama
from colorama import Fore, Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import typer
from typer import Typer, Option
from rich.progress import Progress
from pydantic import BaseModel

class CLIConfig(BaseModel):
    """Configuration for CLI."""
    theme: str = "dark"
    show_progress: bool = True
    show_timestamps: bool = True
    max_table_rows: int = 50
    cache_size: int = 1000
    cache_ttl: int = 3600

class CLIService(cmd.Cmd):
    """Command-line interface service for agentic automation."""
    
    intro = 'Welcome to the CLI. Type help or ? to list commands.\n'
    prompt = '(cli) '
    
    def __init__(self, config_path: str = "config/cli.json"):
        """Initialize CLI service."""
        super().__init__()
        self.config_path = config_path
        self.console = Console()
        self.app = Typer(name="meta-agents", help="Meta-Agents CLI", add_completion=False)
        self.setup_logging()
        self.load_config()
        self.setup_completion()
        colorama.init()
        self._add_commands()
    
    def setup_logging(self) -> None:
        """Set up logging."""
        log_path = Path("logs/cli")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "cli_service.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> None:
        """Load configuration."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.json'):
                    self.config = json.load(f)
                elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
    
    def setup_completion(self) -> None:
        """Set up command completion."""
        try:
            # Set up readline
            if 'libedit' in readline.__doc__:
                readline.parse_and_bind('bind ^I rl_complete')
            else:
                readline.parse_and_bind('tab: complete')
            
            # Set up completion
            self.completekey = 'tab'
            self.cmdqueue = []
        except Exception as e:
            self.logger.error(f"Error setting up completion: {str(e)}")
            raise
    
    def print_error(self, message: str) -> None:
        """Print error message."""
        print(f"{Fore.RED}Error: {message}{Style.RESET_ALL}")
    
    def print_success(self, message: str) -> None:
        """Print success message."""
        print(f"{Fore.GREEN}Success: {message}{Style.RESET_ALL}")
    
    def print_info(self, message: str) -> None:
        """Print info message."""
        print(f"{Fore.BLUE}Info: {message}{Style.RESET_ALL}")
    
    def print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"{Fore.YELLOW}Warning: {message}{Style.RESET_ALL}")
    
    def do_help(self, arg: str) -> None:
        """Show help message."""
        if arg:
            # Show help for specific command
            try:
                func = getattr(self, f'do_{arg}')
                print(func.__doc__ or f'No help available for {arg}')
            except AttributeError:
                self.print_error(f'No help available for {arg}')
        else:
            # Show general help
            print("\nAvailable commands:")
            for name in self.get_names():
                if name.startswith('do_'):
                    command = name[3:]
                    func = getattr(self, name)
                    doc = func.__doc__ or 'No help available'
                    print(f"\n{command}: {doc}")
    
    def do_exit(self, arg: str) -> bool:
        """Exit the CLI."""
        self.print_info("Goodbye!")
        return True
    
    def do_clear(self, arg: str) -> None:
        """Clear the screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def do_cd(self, arg: str) -> None:
        """Change directory."""
        try:
            if not arg:
                self.print_error("Directory path required")
                return
            
            os.chdir(arg)
            self.print_success(f"Changed directory to: {os.getcwd()}")
        except Exception as e:
            self.print_error(f"Error changing directory: {str(e)}")
    
    def do_pwd(self, arg: str) -> None:
        """Print working directory."""
        print(os.getcwd())
    
    def do_ls(self, arg: str) -> None:
        """List directory contents."""
        try:
            path = arg if arg else '.'
            items = os.listdir(path)
            
            # Sort items
            items.sort(key=lambda x: (not os.path.isdir(os.path.join(path, x)), x.lower()))
            
            # Print items
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print(f"{Fore.BLUE}{item}/{Style.RESET_ALL}")
                else:
                    print(item)
        except Exception as e:
            self.print_error(f"Error listing directory: {str(e)}")
    
    def do_cat(self, arg: str) -> None:
        """Display file contents."""
        try:
            if not arg:
                self.print_error("File path required")
                return
            
            with open(arg, 'r') as f:
                print(f.read())
        except Exception as e:
            self.print_error(f"Error reading file: {str(e)}")
    
    def do_mkdir(self, arg: str) -> None:
        """Create directory."""
        try:
            if not arg:
                self.print_error("Directory path required")
                return
            
            os.makedirs(arg, exist_ok=True)
            self.print_success(f"Created directory: {arg}")
        except Exception as e:
            self.print_error(f"Error creating directory: {str(e)}")
    
    def do_rm(self, arg: str) -> None:
        """Remove file or directory."""
        try:
            if not arg:
                self.print_error("Path required")
                return
            
            if os.path.isdir(arg):
                shutil.rmtree(arg)
                self.print_success(f"Removed directory: {arg}")
            else:
                os.remove(arg)
                self.print_success(f"Removed file: {arg}")
        except Exception as e:
            self.print_error(f"Error removing path: {str(e)}")
    
    def do_cp(self, arg: str) -> None:
        """Copy file or directory."""
        try:
            if not arg:
                self.print_error("Source and destination paths required")
                return
            
            args = arg.split()
            if len(args) != 2:
                self.print_error("Source and destination paths required")
                return
            
            source, dest = args
            if os.path.isdir(source):
                shutil.copytree(source, dest)
                self.print_success(f"Copied directory: {source} -> {dest}")
            else:
                shutil.copy2(source, dest)
                self.print_success(f"Copied file: {source} -> {dest}")
        except Exception as e:
            self.print_error(f"Error copying path: {str(e)}")
    
    def do_mv(self, arg: str) -> None:
        """Move file or directory."""
        try:
            if not arg:
                self.print_error("Source and destination paths required")
                return
            
            args = arg.split()
            if len(args) != 2:
                self.print_error("Source and destination paths required")
                return
            
            source, dest = args
            shutil.move(source, dest)
            self.print_success(f"Moved: {source} -> {dest}")
        except Exception as e:
            self.print_error(f"Error moving path: {str(e)}")
    
    def do_run(self, arg: str) -> None:
        """Run a command."""
        try:
            if not arg:
                self.print_error("Command required")
                return
            
            result = subprocess.run(arg, shell=True, capture_output=True, text=True)
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                self.print_error(result.stderr)
            
            if result.returncode == 0:
                self.print_success("Command completed successfully")
            else:
                self.print_error(f"Command failed with exit code {result.returncode}")
        except Exception as e:
            self.print_error(f"Error running command: {str(e)}")
    
    def do_config(self, arg: str) -> None:
        """Show or modify configuration."""
        try:
            if not arg:
                # Show all config
                print(json.dumps(self.config, indent=2))
                return
            
            args = arg.split()
            if len(args) == 1:
                # Show specific config value
                value = self.config
                for key in args[0].split('.'):
                    value = value.get(key)
                print(json.dumps(value, indent=2))
            elif len(args) == 2:
                # Set config value
                keys = args[0].split('.')
                value = self.config
                
                for key in keys[:-1]:
                    if key not in value:
                        value[key] = {}
                    value = value[key]
                
                value[keys[-1]] = args[1]
                
                # Save config
                with open(self.config_path, 'w') as f:
                    if self.config_path.endswith('.json'):
                        json.dump(self.config, f, indent=2)
                    elif self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        yaml.dump(self.config, f, default_flow_style=False)
                
                self.print_success(f"Set config value: {args[0]} = {args[1]}")
            else:
                self.print_error("Invalid arguments")
        except Exception as e:
            self.print_error(f"Error managing config: {str(e)}")
    
    def do_history(self, arg: str) -> None:
        """Show command history."""
        try:
            for i in range(readline.get_current_history_length()):
                print(f"{i + 1}: {readline.get_history_item(i + 1)}")
        except Exception as e:
            self.print_error(f"Error showing history: {str(e)}")
    
    def do_clear_history(self, arg: str) -> None:
        """Clear command history."""
        try:
            readline.clear_history()
            self.print_success("Command history cleared")
        except Exception as e:
            self.print_error(f"Error clearing history: {str(e)}")
    
    def do_echo(self, arg: str) -> None:
        """Echo text."""
        print(arg)
    
    def do_date(self, arg: str) -> None:
        """Show current date and time."""
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    def do_whoami(self, arg: str) -> None:
        """Show current user."""
        print(getpass.getuser())
    
    def do_env(self, arg: str) -> None:
        """Show environment variables."""
        try:
            if not arg:
                # Show all environment variables
                for key, value in os.environ.items():
                    print(f"{key}={value}")
            else:
                # Show specific environment variable
                value = os.environ.get(arg)
                if value is not None:
                    print(value)
                else:
                    self.print_error(f"Environment variable not found: {arg}")
        except Exception as e:
            self.print_error(f"Error showing environment variables: {str(e)}")
    
    def do_setenv(self, arg: str) -> None:
        """Set environment variable."""
        try:
            if not arg:
                self.print_error("Variable name and value required")
                return
            
            args = arg.split('=', 1)
            if len(args) != 2:
                self.print_error("Variable name and value required")
                return
            
            name, value = args
            os.environ[name] = value
            self.print_success(f"Set environment variable: {name}={value}")
        except Exception as e:
            self.print_error(f"Error setting environment variable: {str(e)}")
    
    def do_unsetenv(self, arg: str) -> None:
        """Unset environment variable."""
        try:
            if not arg:
                self.print_error("Variable name required")
                return
            
            if arg in os.environ:
                del os.environ[arg]
                self.print_success(f"Unset environment variable: {arg}")
            else:
                self.print_error(f"Environment variable not found: {arg}")
        except Exception as e:
            self.print_error(f"Error unsetting environment variable: {str(e)}")

    def _add_commands(self):
        """Add CLI commands."""
        
        @self.app.command()
        def tasks(
            status: Optional[str] = Option(None, help="Filter by task status"),
            task_type: Optional[str] = Option(None, help="Filter by task type"),
            limit: int = Option(50, help="Maximum number of tasks to show")
        ):
            """List tasks."""
            try:
                # TODO: Implement task listing logic
                table = Table(title="Tasks", show_header=True, header_style="bold magenta")
                table.add_column("ID", style="dim")
                table.add_column("Name")
                table.add_column("Type")
                table.add_column("Status")
                table.add_column("Created")
                
                # Placeholder data
                table.add_row(
                    "1",
                    "Sample Task",
                    "Analysis",
                    "Pending",
                    datetime.now().isoformat()
                )
                
                self.console.print(table)
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def create_task(
            name: str = Option(..., help="Task name"),
            description: str = Option(..., help="Task description"),
            task_type: str = Option(..., help="Task type"),
            priority: str = Option("normal", help="Task priority")
        ):
            """Create a new task."""
            try:
                # TODO: Implement task creation logic
                self.console.print(
                    Panel(
                        f"Task created successfully:\n"
                        f"Name: {name}\n"
                        f"Type: {task_type}\n"
                        f"Priority: {priority}",
                        title="Success",
                        border_style="green"
                    )
                )
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def workflows(
            status: Optional[str] = Option(None, help="Filter by workflow status"),
            limit: int = Option(50, help="Maximum number of workflows to show")
        ):
            """List workflows."""
            try:
                # TODO: Implement workflow listing logic
                table = Table(title="Workflows", show_header=True, header_style="bold magenta")
                table.add_column("ID", style="dim")
                table.add_column("Name")
                table.add_column("Status")
                table.add_column("Created")
                
                # Placeholder data
                table.add_row(
                    "1",
                    "Sample Workflow",
                    "Active",
                    datetime.now().isoformat()
                )
                
                self.console.print(table)
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def create_workflow(
            name: str = Option(..., help="Workflow name"),
            description: str = Option(..., help="Workflow description")
        ):
            """Create a new workflow."""
            try:
                # TODO: Implement workflow creation logic
                self.console.print(
                    Panel(
                        f"Workflow created successfully:\n"
                        f"Name: {name}\n"
                        f"Description: {description}",
                        title="Success",
                        border_style="green"
                    )
                )
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def metrics():
            """Show system metrics."""
            try:
                # TODO: Implement metrics display logic
                table = Table(title="System Metrics", show_header=True, header_style="bold magenta")
                table.add_column("Metric")
                table.add_column("Value")
                
                # Placeholder data
                table.add_row("CPU Usage", "25%")
                table.add_row("Memory Usage", "1.2GB")
                table.add_row("Active Tasks", "5")
                
                self.console.print(table)
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

        @self.app.command()
        def help():
            """Show help information."""
            self.console.print(
                Panel(
                    Markdown("""
                    # Meta-Agents CLI Help

                    ## Available Commands:
                    - `tasks`: List all tasks
                    - `create-task`: Create a new task
                    - `workflows`: List all workflows
                    - `create-workflow`: Create a new workflow
                    - `metrics`: Show system metrics
                    - `help`: Show this help message

                    Use `--help` with any command for more information.
                    """),
                    title="Help",
                    border_style="blue"
                )
            )

    def execute_cli(self) -> None:
        """Execute the command-line interface.
        
        Starts the interactive CLI session and handles user commands.
        """
        self.app()

def main() -> None:
    """Main entry point for the CLI service."""
    parser = argparse.ArgumentParser(description='Command-line interface')
    parser.add_argument('--config', required=True, help='Path to config file')
    args = parser.parse_args()
    
    try:
        cli = CLIService(args.config)
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logging.error(f"Error running CLI: {str(e)}")
        raise

if __name__ == '__main__':
    main() 