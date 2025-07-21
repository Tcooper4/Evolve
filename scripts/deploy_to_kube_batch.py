#!/usr/bin/env python3
"""
Enhanced Kubernetes Deployment Script

Features:
- Argument parsing for image and namespace configuration
- Environment variable support for configuration
- Proper subprocess error handling and return codes
- Comprehensive logging and error reporting
- Health check validation
- Rollback capabilities
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("deploy_kube.log"),
    ],
)
logger = logging.getLogger(__name__)


class KubernetesDeployer:
    """Enhanced Kubernetes deployment manager."""

    def __init__(
        self,
        namespace: str,
        image: str,
        image_tag: str = "latest",
        registry: str = "your-registry.com",
        timeout: int = 300,
        health_check_timeout: int = 60,
    ):
        self.namespace = namespace
        self.image = image
        self.image_tag = image_tag
        self.registry = registry
        self.timeout = timeout
        self.health_check_timeout = health_check_timeout
        self.full_image_name = f"{registry}/{image}:{image_tag}"

        # Deployment configuration
        self.deployment_name = image
        self.service_name = f"{image}-service"
        self.ingress_name = f"{image}-ingress"

        logger.info(f"Initialized Kubernetes deployer for {self.full_image_name}")

    def run_command(
        self,
        command: List[str],
        capture_output: bool = True,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        try:
            logger.info(f"Running command: {' '.join(command)}")

            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=timeout or self.timeout,
                check=False,  # We'll handle the return code ourselves
            )

            # Log output
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR: {result.stderr}")

            # Handle non-zero return codes
            if result.returncode != 0:
                logger.error(f"Command failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}", file=sys.stderr)
                sys.exit(result.returncode)

            return result

        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {e.timeout} seconds")
            print(f"Command timed out: {' '.join(command)}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError as e:
            logger.error(f"Command not found: {e}")
            print(f"Command not found: {command[0]}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error running command: {e}")
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)

    def check_prerequisites(self) -> bool:
        """Check if kubectl and docker are available."""
        try:
            # Check kubectl
            kubectl_result = self.run_command(
                ["kubectl", "version", "--client"], timeout=30
            )
            logger.info("kubectl is available")

            # Check docker
            docker_result = self.run_command(["docker", "--version"], timeout=30)
            logger.info("docker is available")

            return True

        except Exception as e:
            logger.error(f"Prerequisites check failed: {e}")
            return False

    def create_namespace(self) -> bool:
        """Create namespace if it doesn't exist."""
        try:
            logger.info(f"Creating namespace: {self.namespace}")

            # Check if namespace exists
            check_result = self.run_command(
                ["kubectl", "get", "namespace", self.namespace], timeout=30
            )

            # If namespace doesn't exist, create it
            if check_result.returncode != 0:
                create_result = self.run_command(
                    ["kubectl", "create", "namespace", self.namespace], timeout=30
                )
                logger.info(f"Namespace {self.namespace} created successfully")
            else:
                logger.info(f"Namespace {self.namespace} already exists")

            return True

        except Exception as e:
            logger.error(f"Failed to create namespace: {e}")
            return False

    def build_and_push_image(self) -> bool:
        """Build and push Docker image."""
        try:
            logger.info(f"Building Docker image: {self.full_image_name}")

            # Build image
            build_result = self.run_command(
                ["docker", "build", "-t", self.full_image_name, "."], timeout=600
            )  # 10 minutes for build

            logger.info("Docker image built successfully")

            # Push image
            logger.info(f"Pushing Docker image: {self.full_image_name}")
            push_result = self.run_command(
                ["docker", "push", self.full_image_name], timeout=300
            )  # 5 minutes for push

            logger.info("Docker image pushed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to build/push image: {e}")
            return False

    def update_deployment_config(self) -> bool:
        """Update deployment configuration with new image."""
        try:
            deployment_file = Path("kubernetes/deployment.yaml")

            if not deployment_file.exists():
                logger.error("deployment.yaml not found in kubernetes/ directory")
                return False

            logger.info("Updating deployment configuration")

            # Read current deployment file
            with open(deployment_file, "r") as f:
                content = f.read()

            # Replace image placeholder
            updated_content = content.replace(
                "image: automation:latest", f"image: {self.full_image_name}"
            )

            # Write updated content
            with open(deployment_file, "w") as f:
                f.write(updated_content)

            logger.info("Deployment configuration updated")
            return True

        except Exception as e:
            logger.error(f"Failed to update deployment config: {e}")
            return False

    def apply_kubernetes_configs(self) -> bool:
        """Apply Kubernetes configurations."""
        try:
            logger.info("Applying Kubernetes configurations")

            # Apply deployment
            apply_result = self.run_command(
                [
                    "kubectl",
                    "apply",
                    "-f",
                    "kubernetes/deployment.yaml",
                    "-n",
                    self.namespace,
                ],
                timeout=60,
            )

            logger.info("Kubernetes configurations applied successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to apply Kubernetes configs: {e}")
            return False

    def wait_for_deployment(self) -> bool:
        """Wait for deployment to be ready."""
        try:
            logger.info(f"Waiting for deployment {self.deployment_name} to be ready")

            rollout_result = self.run_command(
                [
                    "kubectl",
                    "rollout",
                    "status",
                    f"deployment/{self.deployment_name}",
                    "-n",
                    self.namespace,
                ],
                timeout=self.timeout,
            )

            logger.info("Deployment is ready")
            return True

        except Exception as e:
            logger.error(f"Deployment failed to become ready: {e}")
            return False

    def check_service_health(self) -> bool:
        """Check service health."""
        try:
            logger.info("Checking service health")

            # Get service URL
            service_url_result = self.run_command(
                [
                    "kubectl",
                    "get",
                    "ingress",
                    self.ingress_name,
                    "-n",
                    self.namespace,
                    "-o",
                    "jsonpath='{.spec.rules[0].host}'",
                ],
                timeout=30,
            )

            if service_url_result.returncode != 0:
                logger.warning("Could not get service URL, skipping health check")
                return True

            service_url = service_url_result.stdout.strip("'")

            if not service_url:
                logger.warning("Service URL is empty, skipping health check")
                return True

            # Perform health check
            health_check_result = self.run_command(
                ["curl", "-s", "-f", f"https://{service_url}/health"],
                timeout=self.health_check_timeout,
            )

            if "healthy" in health_check_result.stdout:
                logger.info("Service health check passed")
                return True
            else:
                logger.error("Service health check failed")
                return False

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def show_deployment_status(self) -> None:
        """Show deployment status."""
        try:
            logger.info("Showing deployment status")

            # Show pods
            self.run_command(
                ["kubectl", "get", "pods", "-n", self.namespace], timeout=30
            )

            # Show services
            self.run_command(
                ["kubectl", "get", "services", "-n", self.namespace], timeout=30
            )

            # Show ingress
            self.run_command(
                ["kubectl", "get", "ingress", "-n", self.namespace], timeout=30
            )

        except Exception as e:
            logger.error(f"Failed to show deployment status: {e}")

    def show_logs(self) -> None:
        """Show application logs."""
        try:
            logger.info("Showing application logs")

            self.run_command(
                [
                    "kubectl",
                    "logs",
                    "-f",
                    f"deployment/{self.deployment_name}",
                    "-n",
                    self.namespace,
                ],
                timeout=60,
                capture_output=False,
            )

        except Exception as e:
            logger.error(f"Failed to show logs: {e}")

    def rollback_deployment(self) -> bool:
        """Rollback deployment if needed."""
        try:
            logger.info("Rolling back deployment")

            rollback_result = self.run_command(
                [
                    "kubectl",
                    "rollout",
                    "undo",
                    f"deployment/{self.deployment_name}",
                    "-n",
                    self.namespace,
                ],
                timeout=60,
            )

            logger.info("Deployment rolled back successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback deployment: {e}")
            return False

    def deploy(self, show_logs: bool = False) -> bool:
        """Execute the complete deployment process."""
        try:
            logger.info("Starting Kubernetes deployment process")

            # Step 1: Check prerequisites
            if not self.check_prerequisites():
                logger.error("Prerequisites check failed")
                return False

            # Step 2: Create namespace
            if not self.create_namespace():
                logger.error("Failed to create namespace")
                return False

            # Step 3: Build and push image
            if not self.build_and_push_image():
                logger.error("Failed to build/push image")
                return False

            # Step 4: Update deployment config
            if not self.update_deployment_config():
                logger.error("Failed to update deployment config")
                return False

            # Step 5: Apply Kubernetes configs
            if not self.apply_kubernetes_configs():
                logger.error("Failed to apply Kubernetes configs")
                return False

            # Step 6: Wait for deployment
            if not self.wait_for_deployment():
                logger.error("Deployment failed to become ready")
                # Attempt rollback
                self.rollback_deployment()
                return False

            # Step 7: Check service health
            if not self.check_service_health():
                logger.error("Service health check failed")
                # Attempt rollback
                self.rollback_deployment()
                return False

            # Step 8: Show deployment status
            self.show_deployment_status()

            # Step 9: Show logs if requested
            if show_logs:
                self.show_logs()

            logger.info("Deployment completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False


def get_config_from_env() -> Dict[str, str]:
    """Get configuration from environment variables."""
    return {
        "namespace": os.getenv("KUBE_NAMESPACE", "automation"),
        "image": os.getenv("KUBE_IMAGE", "automation"),
        "image_tag": os.getenv("KUBE_IMAGE_TAG", "latest"),
        "registry": os.getenv("KUBE_REGISTRY", "your-registry.com"),
        "timeout": os.getenv("KUBE_TIMEOUT", "300"),
        "health_check_timeout": os.getenv("KUBE_HEALTH_TIMEOUT", "60"),
    }


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Kubernetes Deployment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --namespace production --image myapp --tag v1.2.3
  %(prog)s --namespace staging --image frontend --registry my-registry.com
  %(prog)s --show-logs  # Show logs after deployment
        """,
    )

    # Configuration arguments
    parser.add_argument(
        "--namespace",
        "-n",
        help="Kubernetes namespace (default: from KUBE_NAMESPACE env var or 'automation')",
    )
    parser.add_argument(
        "--image",
        "-i",
        help="Docker image name (default: from KUBE_IMAGE env var or 'automation')",
    )
    parser.add_argument(
        "--tag",
        "-t",
        help="Docker image tag (default: from KUBE_IMAGE_TAG env var or 'latest')",
    )
    parser.add_argument(
        "--registry",
        "-r",
        help="Docker registry (default: from KUBE_REGISTRY env var or 'your-registry.com')",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Deployment timeout in seconds (default: from KUBE_TIMEOUT env var or 300)",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        help="Health check timeout in seconds (default: from KUBE_HEALTH_TIMEOUT env var or 60)",
    )

    # Control arguments
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Show application logs after deployment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deployed without actually deploying",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get configuration (command line args override environment variables)
    env_config = get_config_from_env()

    config = {
        "namespace": args.namespace or env_config["namespace"],
        "image": args.image or env_config["image"],
        "image_tag": args.tag or env_config["image_tag"],
        "registry": args.registry or env_config["registry"],
        "timeout": args.timeout or int(env_config["timeout"]),
        "health_check_timeout": args.health_timeout
        or int(env_config["health_check_timeout"]),
    }

    # Validate configuration
    if not config["namespace"] or not config["image"]:
        logger.error("Namespace and image are required")
        parser.print_help()
        sys.exit(1)

    # Log configuration
    logger.info("Deployment configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Dry run mode
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual deployment will be performed")
        logger.info(
            f"Would deploy {config['registry']}/{config['image']}:{config['image_tag']}"
        )
        logger.info(f"to namespace: {config['namespace']}")
        return

    # Create deployer and execute deployment
    deployer = KubernetesDeployer(**config)

    success = deployer.deploy(show_logs=args.show_logs)

    if success:
        logger.info("Deployment completed successfully!")
        print(
            "Kubernetes deployment completed. All services have been "
            "deployed successfully."
        )
        print(f"\nâœ… Deployment successful!")
        print(f"ðŸ“¦ Image: {deployer.full_image_name}")
        print(f"ðŸ—ï¸  Namespace: {deployer.namespace}")
        print(f"ðŸŒ Access the application at your configured ingress URL")
    else:
        logger.error("Deployment failed!")
        print(f"\nâŒ Deployment failed!")
        print(f"ðŸ“‹ Check the logs above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
