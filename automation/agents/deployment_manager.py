import logging
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
from datetime import datetime
import asyncio
from dataclasses import dataclass
import aiohttp
import docker
import kubernetes
import boto3
import paramiko
import yaml
import jinja2
import git
import shutil
import subprocess
import os

@dataclass
class Deployment:
    id: str
    name: str
    type: str
    status: str
    start_time: str
    end_time: Optional[str]
    environment: str
    config: Dict
    logs: List[str]
    artifacts: List[str]

class DeploymentManager:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.deployments: Dict[str, Deployment] = {}
        self.deployment_config = config.get('deployment', {})
        self.docker_client = docker.from_env() if self.deployment_config.get('docker', {}).get('enabled') else None
        self.k8s_client = self._init_kubernetes() if self.deployment_config.get('kubernetes', {}).get('enabled') else None
        self.aws_client = self._init_aws() if self.deployment_config.get('aws', {}).get('enabled') else None
        self.ssh_client = None

    def setup_logging(self):
        """Configure logging for the deployment manager."""
        log_path = Path("automation/logs/deployment")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "deployment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _init_kubernetes(self) -> Optional[kubernetes.client.CoreV1Api]:
        """Initialize Kubernetes client."""
        try:
            kubernetes.config.load_kube_config()
            return kubernetes.client.CoreV1Api()
        except Exception as e:
            self.logger.error(f"Failed to initialize Kubernetes client: {str(e)}")
            return None

    def _init_aws(self) -> Optional[Dict]:
        """Initialize AWS clients."""
        try:
            return {
                'ec2': boto3.client('ec2'),
                'ecs': boto3.client('ecs'),
                's3': boto3.client('s3')
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS clients: {str(e)}")
            return None

    async def deploy(
        self,
        name: str,
        type: str,
        environment: str,
        config: Dict
    ) -> str:
        """
        Deploy an application using the specified strategy.
        
        Args:
            name: Deployment name
            type: Deployment type (docker, kubernetes, aws, ssh)
            environment: Target environment
            config: Deployment configuration
        
        Returns:
            str: Deployment ID
        """
        deployment_id = str(len(self.deployments) + 1)
        
        deployment = Deployment(
            id=deployment_id,
            name=name,
            type=type,
            status='pending',
            start_time=datetime.now().isoformat(),
            end_time=None,
            environment=environment,
            config=config,
            logs=[],
            artifacts=[]
        )
        
        self.deployments[deployment_id] = deployment
        self.logger.info(f"Starting deployment {deployment_id}: {name}")
        
        try:
            if type == 'docker':
                await self._deploy_docker(deployment)
            elif type == 'kubernetes':
                await self._deploy_kubernetes(deployment)
            elif type == 'aws':
                await self._deploy_aws(deployment)
            elif type == 'ssh':
                await self._deploy_ssh(deployment)
            else:
                raise ValueError(f"Unsupported deployment type: {type}")
            
            deployment.status = 'completed'
            deployment.end_time = datetime.now().isoformat()
            self.logger.info(f"Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            deployment.status = 'failed'
            deployment.end_time = datetime.now().isoformat()
            deployment.logs.append(f"Deployment failed: {str(e)}")
            self.logger.error(f"Deployment {deployment_id} failed: {str(e)}")
            raise
        
        return deployment_id

    async def _deploy_docker(self, deployment: Deployment):
        """Deploy using Docker."""
        config = deployment.config
        
        # Build image
        self.logger.info(f"Building Docker image for {deployment.name}")
        image, logs = self.docker_client.images.build(
            path=config['build_path'],
            tag=f"{deployment.name}:{config.get('tag', 'latest')}",
            rm=True
        )
        deployment.logs.extend(logs)
        
        # Push image if registry specified
        if 'registry' in config:
            self.logger.info(f"Pushing image to registry: {config['registry']}")
            self.docker_client.images.push(
                f"{config['registry']}/{deployment.name}:{config.get('tag', 'latest')}"
            )
        
        # Run container
        self.logger.info(f"Starting container for {deployment.name}")
        container = self.docker_client.containers.run(
            image.id,
            detach=True,
            name=deployment.name,
            ports=config.get('ports', {}),
            environment=config.get('environment', {}),
            volumes=config.get('volumes', {}),
            network=config.get('network', 'bridge')
        )
        
        deployment.artifacts.append(container.id)

    async def _deploy_kubernetes(self, deployment: Deployment):
        """Deploy using Kubernetes."""
        config = deployment.config
        
        # Load and render templates
        template_loader = jinja2.FileSystemLoader(config['templates_path'])
        template_env = jinja2.Environment(loader=template_loader)
        
        # Deploy each resource
        for resource in config['resources']:
            template = template_env.get_template(f"{resource}.yaml.j2")
            rendered = template.render(**config['variables'])
            
            # Apply resource
            self.logger.info(f"Applying Kubernetes resource: {resource}")
            subprocess.run(
                ['kubectl', 'apply', '-f', '-'],
                input=rendered.encode(),
                check=True
            )
            
            deployment.artifacts.append(resource)

    async def _deploy_aws(self, deployment: Deployment):
        """Deploy using AWS services."""
        config = deployment.config
        
        if 'ecs' in config:
            # Deploy to ECS
            self.logger.info(f"Deploying to ECS: {deployment.name}")
            response = self.aws_client['ecs'].update_service(
                cluster=config['ecs']['cluster'],
                service=config['ecs']['service'],
                taskDefinition=config['ecs']['task_definition'],
                desiredCount=config['ecs'].get('desired_count', 1)
            )
            deployment.artifacts.append(response['service']['serviceArn'])
        
        elif 'ec2' in config:
            # Deploy to EC2
            self.logger.info(f"Deploying to EC2: {deployment.name}")
            response = self.aws_client['ec2'].run_instances(
                ImageId=config['ec2']['ami'],
                InstanceType=config['ec2']['instance_type'],
                MinCount=1,
                MaxCount=1,
                UserData=config['ec2'].get('user_data', ''),
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [{'Key': 'Name', 'Value': deployment.name}]
                }]
            )
            deployment.artifacts.append(response['Instances'][0]['InstanceId'])

    async def _deploy_ssh(self, deployment: Deployment):
        """Deploy using SSH."""
        config = deployment.config
        
        # Connect to remote server
        self.logger.info(f"Connecting to remote server: {config['host']}")
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(
            hostname=config['host'],
            port=config.get('port', 22),
            username=config['username'],
            key_filename=config.get('key_file')
        )
        
        # Execute deployment commands
        for command in config['commands']:
            self.logger.info(f"Executing command: {command}")
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            deployment.logs.extend(stdout.readlines())
            
            if stderr.channel.recv_exit_status() != 0:
                raise Exception(f"Command failed: {stderr.read().decode()}")

    async def rollback(self, deployment_id: str):
        """Rollback a deployment."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        self.logger.info(f"Rolling back deployment {deployment_id}")
        
        try:
            if deployment.type == 'docker':
                await self._rollback_docker(deployment)
            elif deployment.type == 'kubernetes':
                await self._rollback_kubernetes(deployment)
            elif deployment.type == 'aws':
                await self._rollback_aws(deployment)
            elif deployment.type == 'ssh':
                await self._rollback_ssh(deployment)
            
            deployment.status = 'rolled_back'
            self.logger.info(f"Deployment {deployment_id} rolled back successfully")
            
        except Exception as e:
            self.logger.error(f"Rollback failed for deployment {deployment_id}: {str(e)}")
            raise

    async def _rollback_docker(self, deployment: Deployment):
        """Rollback Docker deployment."""
        # Stop and remove container
        for container_id in deployment.artifacts:
            try:
                container = self.docker_client.containers.get(container_id)
                container.stop()
                container.remove()
            except:
                pass

    async def _rollback_kubernetes(self, deployment: Deployment):
        """Rollback Kubernetes deployment."""
        # Delete resources in reverse order
        for resource in reversed(deployment.artifacts):
            self.logger.info(f"Deleting Kubernetes resource: {resource}")
            subprocess.run(
                ['kubectl', 'delete', '-f', '-'],
                input=resource.encode(),
                check=True
            )

    async def _rollback_aws(self, deployment: Deployment):
        """Rollback AWS deployment."""
        for artifact in deployment.artifacts:
            if artifact.startswith('arn:aws:ecs'):
                # Rollback ECS service
                self.aws_client['ecs'].update_service(
                    cluster=deployment.config['ecs']['cluster'],
                    service=deployment.config['ecs']['service'],
                    taskDefinition=deployment.config['ecs']['previous_task_definition']
                )
            elif artifact.startswith('i-'):
                # Terminate EC2 instance
                self.aws_client['ec2'].terminate_instances(
                    InstanceIds=[artifact]
                )

    async def _rollback_ssh(self, deployment: Deployment):
        """Rollback SSH deployment."""
        if not self.ssh_client:
            return
        
        # Execute rollback commands
        for command in deployment.config.get('rollback_commands', []):
            self.logger.info(f"Executing rollback command: {command}")
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            
            if stderr.channel.recv_exit_status() != 0:
                raise Exception(f"Rollback command failed: {stderr.read().decode()}")

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get deployment details."""
        return self.deployments.get(deployment_id)

    def get_deployments(
        self,
        type: Optional[str] = None,
        status: Optional[str] = None,
        environment: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Deployment]:
        """Get deployments with optional filtering."""
        deployments = list(self.deployments.values())
        
        if type:
            deployments = [d for d in deployments if d.type == type]
        if status:
            deployments = [d for d in deployments if d.status == status]
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        
        # Sort by start time (newest first)
        deployments.sort(key=lambda x: x.start_time, reverse=True)
        
        if limit:
            deployments = deployments[:limit]
        
        return deployments

    def clear_deployments(self, before: Optional[datetime] = None):
        """Clear old deployments."""
        if before:
            self.deployments = {
                id: d for id, d in self.deployments.items()
                if datetime.fromisoformat(d.start_time) > before
            }
        else:
            self.deployments.clear()
        
        self.logger.info(f"Cleared deployments before {before}")

    async def verify_deployment(self, deployment_id: str) -> bool:
        """Verify a deployment's success."""
        deployment = self.get_deployment(deployment_id)
        if not deployment:
            return False
        
        try:
            if deployment.type == 'docker':
                return await self._verify_docker(deployment)
            elif deployment.type == 'kubernetes':
                return await self._verify_kubernetes(deployment)
            elif deployment.type == 'aws':
                return await self._verify_aws(deployment)
            elif deployment.type == 'ssh':
                return await self._verify_ssh(deployment)
            return False
        except Exception as e:
            self.logger.error(f"Deployment verification failed: {str(e)}")
            return False

    async def _verify_docker(self, deployment: Deployment) -> bool:
        """Verify Docker deployment."""
        for container_id in deployment.artifacts:
            try:
                container = self.docker_client.containers.get(container_id)
                if container.status != 'running':
                    return False
                
                # Check health if healthcheck configured
                if container.attrs['State'].get('Health'):
                    if container.attrs['State']['Health']['Status'] != 'healthy':
                        return False
            except:
                return False
        return True

    async def _verify_kubernetes(self, deployment: Deployment) -> bool:
        """Verify Kubernetes deployment."""
        for resource in deployment.artifacts:
            try:
                result = subprocess.run(
                    ['kubectl', 'get', resource, '-o', 'json'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                status = json.loads(result.stdout)
                
                # Check resource status
                if status.get('status', {}).get('phase') != 'Running':
                    return False
            except:
                return False
        return True

    async def _verify_aws(self, deployment: Deployment) -> bool:
        """Verify AWS deployment."""
        for artifact in deployment.artifacts:
            if artifact.startswith('arn:aws:ecs'):
                # Check ECS service
                service = self.aws_client['ecs'].describe_services(
                    cluster=deployment.config['ecs']['cluster'],
                    services=[deployment.config['ecs']['service']]
                )
                if service['services'][0]['status'] != 'ACTIVE':
                    return False
            elif artifact.startswith('i-'):
                # Check EC2 instance
                instance = self.aws_client['ec2'].describe_instances(
                    InstanceIds=[artifact]
                )
                if instance['Reservations'][0]['Instances'][0]['State']['Name'] != 'running':
                    return False
        return True

    async def _verify_ssh(self, deployment: Deployment) -> bool:
        """Verify SSH deployment."""
        if not self.ssh_client:
            return False
        
        for command in deployment.config.get('verify_commands', []):
            try:
                stdin, stdout, stderr = self.ssh_client.exec_command(command)
                if stderr.channel.recv_exit_status() != 0:
                    return False
            except:
                return False
        return True 