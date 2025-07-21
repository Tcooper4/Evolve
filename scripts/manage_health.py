#!/usr/bin/env python3
"""
Health management script.
Provides commands for checking and reporting application and system health.

This script supports:
- Checking application health
- Generating health reports
- Exporting health data

Usage:
    python manage_health.py <command> [options]

Commands:
    check       Check application health
    report      Generate health report
    export      Export health data

Examples:
    # Check application health
    python manage_health.py check

    # Generate health report
    python manage_health.py report --output health_report.json

    # Export health data
    python manage_health.py export --output health_data.csv
"""

import argparse
import asyncio
import json
import logging
import logging.config
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

from utils.launch_utils import setup_logging


class HealthManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the health manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.health_dir = Path("health")
        self.health_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Set up logging for the service."""
        return setup_logging(service_name="report_service")

    def collect_metrics(self, duration: int = 300):
        """Collect system health metrics."""
        self.logger.info(f"Collecting health metrics for {duration} seconds")

        try:
            # Collect system metrics
            system_metrics = self._collect_system_metrics()

            # Collect application metrics
            app_metrics = self._collect_app_metrics()

            # Collect database metrics
            db_metrics = self._collect_db_metrics()

            # Combine metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": system_metrics,
                "application": app_metrics,
                "database": db_metrics,
            }

            # Save metrics
            self._save_metrics(metrics)

            self.logger.info("Health metrics collected successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to collect health metrics: {e}")
            return False

    def generate_dashboard(self, metrics_file: str):
        """Generate health dashboard."""
        self.logger.info(f"Generating dashboard from {metrics_file}")

        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Create dashboard
            app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

            # Define layout
            app.layout = dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H1(
                                        "System Health Dashboard",
                                        className="text-center mb-4",
                                    ),
                                    html.Div(id="last-update"),
                                ]
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Graph(id="cpu-graph"),
                                    dcc.Graph(id="memory-graph"),
                                    dcc.Graph(id="disk-graph"),
                                    dcc.Graph(id="network-graph"),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Graph(id="process-graph"),
                                    dcc.Graph(id="container-graph"),
                                    dcc.Graph(id="kubernetes-graph"),
                                ],
                                width=6,
                            ),
                        ]
                    ),
                    dcc.Interval(
                        id="interval-component", interval=5 * 1000, n_intervals=0
                    ),  # Update every 5 seconds
                ]
            )

            # Define callbacks
            @app.callback(
                [
                    Output("cpu-graph", "figure"),
                    Output("memory-graph", "figure"),
                    Output("disk-graph", "figure"),
                    Output("network-graph", "figure"),
                    Output("process-graph", "figure"),
                    Output("container-graph", "figure"),
                    Output("kubernetes-graph", "figure"),
                    Output("last-update", "children"),
                ],
                [Input("interval-component", "n_intervals")],
            )
            def update_graphs(n):
                # CPU usage
                cpu_fig = go.Figure()
                cpu_fig.add_trace(
                    go.Scatter(
                        x=metrics["timestamp"],
                        y=metrics["cpu_percent"],
                        name="CPU Usage",
                    )
                )
                cpu_fig.update_layout(
                    title="CPU Usage", xaxis_title="Time", yaxis_title="Usage (%)"
                )

                # Memory usage
                memory_fig = go.Figure()
                memory_fig.add_trace(
                    go.Scatter(
                        x=metrics["timestamp"],
                        y=metrics["memory_percent"],
                        name="Memory Usage",
                    )
                )
                memory_fig.update_layout(
                    title="Memory Usage", xaxis_title="Time", yaxis_title="Usage (%)"
                )

                # Disk usage
                disk_fig = go.Figure()
                disk_fig.add_trace(
                    go.Scatter(
                        x=metrics["timestamp"],
                        y=metrics["disk_usage"],
                        name="Disk Usage",
                    )
                )
                disk_fig.update_layout(
                    title="Disk Usage", xaxis_title="Time", yaxis_title="Usage (%)"
                )

                # Network I/O
                network_fig = go.Figure()
                network_fig.add_trace(
                    go.Scatter(
                        x=metrics["timestamp"],
                        y=[m["bytes_sent"] for m in metrics["network_io"]],
                        name="Bytes Sent",
                    )
                )
                network_fig.add_trace(
                    go.Scatter(
                        x=metrics["timestamp"],
                        y=[m["bytes_recv"] for m in metrics["network_io"]],
                        name="Bytes Received",
                    )
                )
                network_fig.update_layout(
                    title="Network I/O", xaxis_title="Time", yaxis_title="Bytes"
                )

                # Process metrics
                process_df = pd.DataFrame(
                    [
                        {
                            "timestamp": t,
                            "pid": p["pid"],
                            "name": p["name"],
                            "cpu_percent": p["cpu_percent"],
                            "memory_percent": p["memory_percent"],
                        }
                        for t, processes in zip(
                            metrics["timestamp"], metrics["process_metrics"]
                        )
                        for p in processes
                    ]
                )
                process_fig = px.scatter(
                    process_df,
                    x="timestamp",
                    y="cpu_percent",
                    color="name",
                    title="Process CPU Usage",
                )

                # Container metrics
                container_df = pd.DataFrame(
                    [
                        {
                            "timestamp": t,
                            "name": c["name"],
                            "cpu_usage": c["cpu_usage"],
                            "memory_usage": c["memory_usage"],
                        }
                        for t, containers in zip(
                            metrics["timestamp"], metrics["container_metrics"]
                        )
                        for c in containers
                    ]
                )
                container_fig = px.scatter(
                    container_df,
                    x="timestamp",
                    y="cpu_usage",
                    color="name",
                    title="Container CPU Usage",
                )

                # Kubernetes metrics
                k8s_df = pd.DataFrame(
                    [
                        {
                            "timestamp": t,
                            "name": p["name"],
                            "namespace": p["namespace"],
                            "status": p["status"],
                            "containers": len(p["containers"]),
                            "ready_containers": sum(
                                1 for c in p["containers"] if c["ready"]
                            ),
                        }
                        for t, pods in zip(
                            metrics["timestamp"], metrics["kubernetes_metrics"]
                        )
                        for p in pods
                    ]
                )
                k8s_fig = px.scatter(
                    k8s_df,
                    x="timestamp",
                    y="ready_containers",
                    color="name",
                    title="Kubernetes Pod Status",
                )

                last_update = (
                    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

                return (
                    cpu_fig,
                    memory_fig,
                    disk_fig,
                    network_fig,
                    process_fig,
                    container_fig,
                    k8s_fig,
                    last_update,
                )

            # Save dashboard
            dashboard_file = self.dashboard_dir / "dashboard.py"
            with open(dashboard_file, "w") as f:
                f.write(app.to_string())

            self.logger.info(f"Dashboard saved to {dashboard_file}")
            return dashboard_file
        except Exception as e:
            self.logger.error(f"Failed to generate dashboard: {e}")
            raise

    def run_dashboard(self, dashboard_file: str):
        """Run the health dashboard."""
        self.logger.info(f"Running dashboard from {dashboard_file}")

        try:
            # Run dashboard
            subprocess.run(["python", dashboard_file])
        except Exception as e:
            self.logger.error(f"Failed to run dashboard: {e}")
            raise

    def analyze_health(self, metrics_file: str):
        """Analyze system health."""
        self.logger.info(f"Analyzing health from {metrics_file}")

        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Calculate statistics
            stats = {
                "cpu": {
                    "mean": sum(metrics["cpu_percent"]) / len(metrics["cpu_percent"]),
                    "max": max(metrics["cpu_percent"]),
                    "min": min(metrics["cpu_percent"]),
                },
                "memory": {
                    "mean": sum(metrics["memory_percent"])
                    / len(metrics["memory_percent"]),
                    "max": max(metrics["memory_percent"]),
                    "min": min(metrics["memory_percent"]),
                },
                "disk": {
                    "mean": sum(metrics["disk_usage"]) / len(metrics["disk_usage"]),
                    "max": max(metrics["disk_usage"]),
                    "min": min(metrics["disk_usage"]),
                },
            }

            # Generate recommendations
            recommendations = []

            if stats["cpu"]["mean"] > 80:
                recommendations.append(
                    "High CPU usage detected. Consider scaling up CPU resources."
                )
            if stats["memory"]["mean"] > 80:
                recommendations.append(
                    "High memory usage detected. Consider increasing memory allocation."
                )
            if stats["disk"]["mean"] > 80:
                recommendations.append(
                    "High disk usage detected. Consider cleaning up disk space."
                )

            # Save analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = self.metrics_dir / f"analysis_{timestamp}.json"

            analysis = {
                "timestamp": timestamp,
                "statistics": stats,
                "recommendations": recommendations,
            }

            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)

            self.logger.info(f"Analysis saved to {analysis_file}")
            return analysis
        except Exception as e:
            self.logger.error(f"Failed to analyze health: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Health Manager")
    parser.add_argument(
        "command",
        choices=["collect", "dashboard", "run", "analyze"],
        help="Command to execute",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration for metrics collection in seconds",
    )
    parser.add_argument("--metrics-file", help="Metrics file to use")
    parser.add_argument("--dashboard-file", help="Dashboard file to use")

    args = parser.parse_args()
    manager = HealthManager()

    commands = {
        "collect": lambda: asyncio.run(manager.collect_metrics(args.duration)),
        "dashboard": lambda: manager.generate_dashboard(args.metrics_file),
        "run": lambda: manager.run_dashboard(args.dashboard_file),
        "analyze": lambda: manager.analyze_health(args.metrics_file),
    }

    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
