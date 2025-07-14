#!/usr/bin/env python3
"""
Dashboard management script.
Provides commands for managing the application dashboard, including launching, updating, and monitoring dashboard status.

This script supports:
- Launching the dashboard
- Updating dashboard components
- Monitoring dashboard status

Usage:
    python manage_dashboard.py <command> [options]

Commands:
    launch      Launch the dashboard
    update      Update dashboard components
    status      Monitor dashboard status

Examples:
    # Launch the dashboard
    python manage_dashboard.py launch

    # Update dashboard components
    python manage_dashboard.py update

    # Monitor dashboard status
    python manage_dashboard.py status
"""

import argparse
import json
import logging
import logging.config
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import yaml
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

import docker
from kubernetes import client, config


class DashboardManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the dashboard manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.dashboard_dir = Path("dashboard")
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    def setup_logging(self):
        """Initialize logging configuration."""
        log_config_path = Path("config/logging_config.yaml")
        if not log_config_path.exists():
            print("Error: logging_config.yaml not found")
            sys.exit(1)

        with open(log_config_path) as f:
            log_config = yaml.safe_load(f)

        logging.config.dictConfig(log_config)

    def create_dashboard(self, dashboard_type: str = "system"):
        """Create a monitoring dashboard."""
        self.logger.info(f"Creating {dashboard_type} dashboard")

        try:
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
                                        "System Monitoring Dashboard",
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
                        x=pd.date_range(
                            start=datetime.now() - timedelta(minutes=5),
                            periods=60,
                            freq="5S",
                        ),
                        y=[psutil.cpu_percent() for _ in range(60)],
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
                        x=pd.date_range(
                            start=datetime.now() - timedelta(minutes=5),
                            periods=60,
                            freq="5S",
                        ),
                        y=[psutil.virtual_memory().percent for _ in range(60)],
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
                        x=pd.date_range(
                            start=datetime.now() - timedelta(minutes=5),
                            periods=60,
                            freq="5S",
                        ),
                        y=[psutil.disk_usage("/").percent for _ in range(60)],
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
                        x=pd.date_range(
                            start=datetime.now() - timedelta(minutes=5),
                            periods=60,
                            freq="5S",
                        ),
                        y=[psutil.net_io_counters().bytes_sent for _ in range(60)],
                        name="Bytes Sent",
                    )
                )
                network_fig.add_trace(
                    go.Scatter(
                        x=pd.date_range(
                            start=datetime.now() - timedelta(minutes=5),
                            periods=60,
                            freq="5S",
                        ),
                        y=[psutil.net_io_counters().bytes_recv for _ in range(60)],
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
                            "timestamp": datetime.now(),
                            "pid": proc.pid,
                            "name": proc.name(),
                            "cpu_percent": proc.cpu_percent(),
                            "memory_percent": proc.memory_percent(),
                        }
                        for proc in psutil.process_iter(
                            ["pid", "name", "cpu_percent", "memory_percent"]
                        )
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
                try:
                    docker_client = docker.from_env()
                    container_df = pd.DataFrame(
                        [
                            {
                                "timestamp": datetime.now(),
                                "name": container.name,
                                "cpu_usage": container.stats(stream=False)["cpu_stats"][
                                    "cpu_usage"
                                ]["total_usage"],
                                "memory_usage": container.stats(stream=False)[
                                    "memory_stats"
                                ]["usage"],
                            }
                            for container in docker_client.containers.list()
                        ]
                    )
                    container_fig = px.scatter(
                        container_df,
                        x="timestamp",
                        y="cpu_usage",
                        color="name",
                        title="Container CPU Usage",
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to collect container metrics: {e}")
                    container_fig = go.Figure()

                # Kubernetes metrics
                try:
                    config.load_kube_config()
                    v1 = client.CoreV1Api()
                    k8s_df = pd.DataFrame(
                        [
                            {
                                "timestamp": datetime.now(),
                                "name": pod.metadata.name,
                                "namespace": pod.metadata.namespace,
                                "status": pod.status.phase,
                                "containers": len(pod.status.container_statuses),
                                "ready_containers": sum(
                                    1 for c in pod.status.container_statuses if c.ready
                                ),
                            }
                            for pod in v1.list_pod_for_all_namespaces().items
                        ]
                    )
                    k8s_fig = px.scatter(
                        k8s_df,
                        x="timestamp",
                        y="ready_containers",
                        color="name",
                        title="Kubernetes Pod Status",
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to collect Kubernetes metrics: {e}")
                    k8s_fig = go.Figure()

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
            dashboard_file = self.dashboard_dir / f"dashboard_{dashboard_type}.py"
            with open(dashboard_file, "w") as f:
                f.write(app.to_string())

            self.logger.info(f"Dashboard saved to {dashboard_file}")
            return dashboard_file
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            raise

    def run_dashboard(self, dashboard_file: str):
        """Run the monitoring dashboard."""
        self.logger.info(f"Running dashboard from {dashboard_file}")

        try:
            # Run dashboard
            subprocess.run(["python", dashboard_file])
        except Exception as e:
            self.logger.error(f"Failed to run dashboard: {e}")
            raise

    def analyze_metrics(self, metrics_file: str):
        """Analyze system metrics."""
        self.logger.info(f"Analyzing metrics from {metrics_file}")

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
            self.logger.error(f"Failed to analyze metrics: {e}")
            raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Dashboard Manager")
    parser.add_argument(
        "command", choices=["create", "run", "analyze"], help="Command to execute"
    )
    parser.add_argument(
        "--dashboard-type", default="system", help="Type of dashboard to create"
    )
    parser.add_argument("--dashboard-file", help="Dashboard file to use")
    parser.add_argument("--metrics-file", help="Metrics file to use")

    args = parser.parse_args()
    manager = DashboardManager()

    commands = {
        "create": lambda: manager.create_dashboard(args.dashboard_type),
        "run": lambda: manager.run_dashboard(args.dashboard_file),
        "analyze": lambda: manager.analyze_metrics(args.metrics_file),
    }

    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
