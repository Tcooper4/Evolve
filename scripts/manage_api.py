#!/usr/bin/env python3
"""
API management script.
Provides commands for managing the application's API server, including starting, stopping, and checking status.

This script supports:
- Starting the API server
- Stopping the API server
- Checking API server status

Usage:
    python manage_api.py <command> [options]

Commands:
    start       Start the API server
    stop        Stop the API server
    status      Check API server status

Examples:
    # Start the API server
    python manage_api.py start

    # Stop the API server
    python manage_api.py stop

    # Check API server status
    python manage_api.py status
"""

import argparse
import asyncio
import json
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import markdown
import requests
import yaml
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from jinja2 import Environment, FileSystemLoader


class APIManager:
    def __init__(self, config_path: str = "config/app_config.yaml"):
        """Initialize the API manager."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.logger = logging.getLogger("trading")
        self.api_dir = Path("api")
        self.api_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir = Path("docs/api")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir = Path("tests/api")
        self.tests_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load application configuration."""
        if not Path(config_path).exists():
            print(f"Error: Configuration file not found: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            return yaml.safe_load(f)

    from utils.launch_utils import setup_logging

def setup_logging():
    """Set up logging for the service."""
    return setup_logging(service_name="service")def generate_openapi(self, app: FastAPI, version: str = "1.0.0"):
        """Generate OpenAPI specification."""
        self.logger.info(f"Generating OpenAPI specification for version {version}")

        try:
            # Get OpenAPI schema
            openapi_schema = get_openapi(
                title=app.title,
                version=version,
                description=app.description,
                routes=app.routes,
            )

            # Save schema
            schema_file = self.api_dir / f"openapi_{version}.json"
            with open(schema_file, "w") as f:
                json.dump(openapi_schema, f, indent=2)

            self.logger.info(f"OpenAPI specification saved to {schema_file}")
            return openapi_schema
        except Exception as e:
            self.logger.error(f"Failed to generate OpenAPI specification: {e}")
            raise

    def generate_docs(self, openapi_schema: Dict[str, Any], version: str = "1.0.0"):
        """Generate API documentation."""
        self.logger.info(f"Generating API documentation for version {version}")

        try:
            # Load templates
            env = Environment(loader=FileSystemLoader("templates/api"))
            template = env.get_template("api_docs.md.j2")

            # Generate documentation
            docs = template.render(
                version=version,
                openapi=openapi_schema,
                timestamp=datetime.now().isoformat(),
            )

            # Save documentation
            docs_file = self.docs_dir / f"api_docs_{version}.md"
            with open(docs_file, "w") as f:
                f.write(docs)

            # Convert to HTML
            html = markdown.markdown(docs)
            html_file = self.docs_dir / f"api_docs_{version}.html"
            with open(html_file, "w") as f:
                f.write(html)

            self.logger.info(f"API documentation saved to {docs_file} and {html_file}")
            return docs_file, html_file
        except Exception as e:
            self.logger.error(f"Failed to generate API documentation: {e}")
            raise

    def generate_tests(self, openapi_schema: Dict[str, Any], version: str = "1.0.0"):
        """Generate API tests."""
        self.logger.info(f"Generating API tests for version {version}")

        try:
            # Load templates
            env = Environment(loader=FileSystemLoader("templates/api"))
            template = env.get_template("api_tests.py.j2")

            # Generate tests
            tests = template.render(
                version=version,
                openapi=openapi_schema,
                timestamp=datetime.now().isoformat(),
            )

            # Save tests
            tests_file = self.tests_dir / f"test_api_{version}.py"
            with open(tests_file, "w") as f:
                f.write(tests)

            self.logger.info(f"API tests saved to {tests_file}")
            return tests_file
        except Exception as e:
            self.logger.error(f"Failed to generate API tests: {e}")
            raise

    async def test_endpoints(self, base_url: str, version: str = "1.0.0"):
        """Test API endpoints."""
        self.logger.info(f"Testing API endpoints for version {version}")

        try:
            # Load OpenAPI schema
            schema_file = self.api_dir / f"openapi_{version}.json"
            with open(schema_file) as f:
                openapi_schema = json.load(f)

            # Test endpoints
            async with aiohttp.ClientSession() as session:
                results = []
                for path, path_item in openapi_schema["paths"].items():
                    for method, operation in path_item.items():
                        if method in ["get", "post", "put", "delete"]:
                            url = f"{base_url}{path}"
                            try:
                                async with session.request(method, url) as response:
                                    results.append(
                                        {
                                            "endpoint": f"{method.upper()} {path}",
                                            "status": response.status,
                                            "success": 200 <= response.status < 300,
                                        }
                                    )
                            except Exception as e:
                                results.append(
                                    {
                                        "endpoint": f"{method.upper()} {path}",
                                        "error": str(e),
                                        "success": False,
                                    }
                                )

            # Save results
            results_file = self.api_dir / f"test_results_{version}.json"
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "version": version,
                        "results": results,
                    },
                    f,
                    indent=2,
                )

            # Print results
            self._print_test_results(results)

            return all(result["success"] for result in results)
        except Exception as e:
            self.logger.error(f"Failed to test endpoints: {e}")
            raise

    def monitor_endpoints(
        self, base_url: str, version: str = "1.0.0", duration: int = 300
    ):
        """Monitor API endpoints."""
        self.logger.info(f"Monitoring API endpoints for version {version}")

        try:
            # Load OpenAPI schema
            schema_file = self.api_dir / f"openapi_{version}.json"
            with open(schema_file) as f:
                openapi_schema = json.load(f)

            # Monitor endpoints
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=duration)

            metrics = []
            while datetime.now() < end_time:
                for path, path_item in openapi_schema["paths"].items():
                    for method, operation in path_item.items():
                        if method in ["get", "post", "put", "delete"]:
                            url = f"{base_url}{path}"
                            try:
                                start = time.time()
                                response = requests.request(method, url)
                                latency = time.time() - start

                                metrics.append(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "endpoint": f"{method.upper()} {path}",
                                        "status": response.status,
                                        "latency": latency,
                                    }
                                )
                            except Exception as e:
                                metrics.append(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "endpoint": f"{method.upper()} {path}",
                                        "error": str(e),
                                    }
                                )

                time.sleep(1)  # Wait 1 second between iterations

            # Save metrics
            metrics_file = self.api_dir / f"monitoring_{version}.json"
            with open(metrics_file, "w") as f:
                json.dump(
                    {
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "version": version,
                        "metrics": metrics,
                    },
                    f,
                    indent=2,
                )

            # Print metrics
            self._print_monitoring_metrics(metrics)

            return True
        except Exception as e:
            self.logger.error(f"Failed to monitor endpoints: {e}")
            raise

    def _print_test_results(self, results: List[Dict[str, Any]]):
        """Print test results."""
        print("\nAPI Test Results:")
        print("\nEndpoints:")
        for result in results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['endpoint']}")
            if not result["success"]:
                print(f"  Error: {result.get('error', 'Unknown error')}")

    def _print_monitoring_metrics(self, metrics: List[Dict[str, Any]]):
        """Print monitoring metrics."""
        print("\nAPI Monitoring Metrics:")

        # Group metrics by endpoint
        endpoint_metrics = {}
        for metric in metrics:
            endpoint = metric["endpoint"]
            if endpoint not in endpoint_metrics:
                endpoint_metrics[endpoint] = []
            endpoint_metrics[endpoint].append(metric)

        # Print metrics for each endpoint
        for endpoint, endpoint_data in endpoint_metrics.items():
            print(f"\n{endpoint}:")

            # Calculate statistics
            latencies = [m["latency"] for m in endpoint_data if "latency" in m]
            if latencies:
                print(f"  Average Latency: {sum(latencies) / len(latencies):.3f}s")
                print(f"  Min Latency: {min(latencies):.3f}s")
                print(f"  Max Latency: {max(latencies):.3f}s")

            # Count errors
            errors = [m for m in endpoint_data if "error" in m]
            if errors:
                print(f"  Errors: {len(errors)}")
                for error in errors:
                    print(f"    - {error['error']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="API Manager")
    parser.add_argument(
        "command", choices=["generate", "test", "monitor"], help="Command to execute"
    )
    parser.add_argument("--version", default="1.0.0", help="API version")
    parser.add_argument("--base-url", help="Base URL for API testing and monitoring")
    parser.add_argument(
        "--duration", type=int, default=300, help="Duration for monitoring in seconds"
    )

    args = parser.parse_args()
    manager = APIManager()

    commands = {
        "generate": lambda: manager.generate_docs(
            manager.generate_openapi(FastAPI(), args.version), args.version
        ),
        "test": lambda: asyncio.run(
            manager.test_endpoints(args.base_url, args.version)
        ),
        "monitor": lambda: manager.monitor_endpoints(
            args.base_url, args.version, args.duration
        ),
    }

    if args.command in commands:
        success = commands[args.command]()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
