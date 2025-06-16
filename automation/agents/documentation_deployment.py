import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import re
from datetime import datetime
import json
import yaml
import asyncio
import aiohttp
from dataclasses import dataclass
import docker
import kubernetes
from kubernetes import client, config
import boto3
import google.cloud.storage
import azure.storage.blob
import paramiko
import fabric
import ansible.playbook
import ansible.inventory
import ansible.parsing.dataloader
import ansible.inventory.manager
import ansible.playbook.play
import ansible.executor.task_queue_manager
import ansible.plugins.callback
import ansible.executor.playbook_executor
import ansible.inventory.host
import ansible.vars.manager
import ansible.parsing.dataloader
import ansible.inventory.manager
import ansible.playbook.play
import ansible.executor.task_queue_manager
import ansible.plugins.callback
import ansible.executor.playbook_executor
import ansible.inventory.host
import ansible.vars.manager

@dataclass
class Deployment:
    id: str
    type: str
    status: str
    start_time: str
    end_time: Optional[str]
    config: Dict
    logs: List[str]
    artifacts: List[str]

class DocumentationDeployment:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.deploy_config = config.get('documentation', {}).get('deployment', {})
        self.setup_clients()
        self.deployments: Dict[str, Deployment] = {}

    def setup_logging(self):
        """Configure logging for the documentation deployment system."""
        log_path = Path("automation/logs/documentation")
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

    def setup_clients(self):
        """Setup deployment clients based on configuration."""
        # Docker
        if self.deploy_config.get('docker', {}).get('enabled', False):
            self.docker_client = docker.from_env()
        
        # Kubernetes
        if self.deploy_config.get('kubernetes', {}).get('enabled', False):
            try:
                config.load_kube_config()
                self.k8s_client = client.CoreV1Api()
            except Exception as e:
                self.logger.warning(f"Failed to load Kubernetes config: {str(e)}")
        
        # AWS
        if self.deploy_config.get('aws', {}).get('enabled', False):
            self.aws_client = boto3.client('s3')
        
        # GCS
        if self.deploy_config.get('gcs', {}).get('enabled', False):
            self.gcs_client = google.cloud.storage.Client()
        
        # Azure
        if self.deploy_config.get('azure', {}).get('enabled', False):
            self.azure_client = azure.storage.blob.BlobServiceClient.from_connection_string(
                self.deploy_config.get('azure', {}).get('connection_string', '')
            )

    async def deploy_docker(
        self,
        image_name: str,
        container_name: str,
        ports: Dict[str, str],
        volumes: Dict[str, str],
        environment: Dict[str, str]
    ) -> Deployment:
        """Deploy documentation using Docker."""
        try:
            # Generate deployment ID
            deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment record
            deployment = Deployment(
                id=deployment_id,
                type='docker',
                status='in_progress',
                start_time=datetime.now().isoformat(),
                end_time=None,
                config={
                    'image_name': image_name,
                    'container_name': container_name,
                    'ports': ports,
                    'volumes': volumes,
                    'environment': environment
                },
                logs=[],
                artifacts=[]
            )
            
            self.deployments[deployment_id] = deployment
            
            try:
                # Run Docker operations in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Pull image
                self.logger.info(f"Pulling image: {image_name}")
                await loop.run_in_executor(None, self.docker_client.images.pull, image_name)
                
                # Create container
                self.logger.info(f"Creating container: {container_name}")
                container = await loop.run_in_executor(
                    None,
                    lambda: self.docker_client.containers.run(
                        image_name,
                        name=container_name,
                        ports=ports,
                        volumes=volumes,
                        environment=environment,
                        detach=True
                    )
                )
                
                # Update deployment
                deployment.status = 'completed'
                deployment.end_time = datetime.now().isoformat()
                deployment.artifacts.append(container.id)
                
                self.logger.info(f"Deployed container: {container_name}")
                return deployment
                
            except Exception as e:
                deployment.status = 'failed'
                deployment.end_time = datetime.now().isoformat()
                deployment.logs.append(f"Deployment failed: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to deploy Docker container: {str(e)}")
            raise

    async def deploy_kubernetes(
        self,
        namespace: str,
        deployment_name: str,
        image: str,
        replicas: int,
        ports: List[Dict],
        volumes: List[Dict],
        environment: Dict[str, str]
    ) -> Deployment:
        """Deploy documentation using Kubernetes."""
        try:
            # Generate deployment ID
            deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment record
            deployment = Deployment(
                id=deployment_id,
                type='kubernetes',
                status='in_progress',
                start_time=datetime.now().isoformat(),
                end_time=None,
                config={
                    'namespace': namespace,
                    'deployment_name': deployment_name,
                    'image': image,
                    'replicas': replicas,
                    'ports': ports,
                    'volumes': volumes,
                    'environment': environment
                },
                logs=[],
                artifacts=[]
            )
            
            self.deployments[deployment_id] = deployment
            
            try:
                # Run Kubernetes operations in a thread pool
                loop = asyncio.get_event_loop()
                
                # Create deployment
                self.logger.info(f"Creating deployment: {deployment_name}")
                k8s_deployment = client.V1Deployment(
                    metadata=client.V1ObjectMeta(name=deployment_name),
                    spec=client.V1DeploymentSpec(
                        replicas=replicas,
                        selector=client.V1LabelSelector(
                            match_labels={'app': deployment_name}
                        ),
                        template=client.V1PodTemplateSpec(
                            metadata=client.V1ObjectMeta(
                                labels={'app': deployment_name}
                            ),
                            spec=client.V1PodSpec(
                                containers=[
                                    client.V1Container(
                                        name=deployment_name,
                                        image=image,
                                        ports=[
                                            client.V1ContainerPort(
                                                container_port=p['container_port'],
                                                name=p.get('name', '')
                                            ) for p in ports
                                        ],
                                        env=[
                                            client.V1EnvVar(
                                                name=k,
                                                value=v
                                            ) for k, v in environment.items()
                                        ],
                                        volume_mounts=[
                                            client.V1VolumeMount(
                                                name=v['name'],
                                                mount_path=v['mount_path']
                                            ) for v in volumes
                                        ]
                                    )
                                ],
                                volumes=[
                                    client.V1Volume(
                                        name=v['name'],
                                        host_path=client.V1HostPathVolumeSource(
                                            path=v['host_path']
                                        )
                                    ) for v in volumes
                                ]
                            )
                        )
                    )
                )
                
                # Apply deployment
                await loop.run_in_executor(
                    None,
                    lambda: self.k8s_client.create_namespaced_deployment(
                        namespace=namespace,
                        body=k8s_deployment
                    )
                )
                
                # Wait for deployment to be ready
                while True:
                    deployment_status = await loop.run_in_executor(
                        None,
                        lambda: self.k8s_client.read_namespaced_deployment_status(
                            name=deployment_name,
                            namespace=namespace
                        )
                    )
                    
                    if deployment_status.status.ready_replicas == replicas:
                        break
                        
                    await asyncio.sleep(1)
                
                # Update deployment
                deployment.status = 'completed'
                deployment.end_time = datetime.now().isoformat()
                deployment.artifacts.append(deployment_name)
                
                self.logger.info(f"Deployed to Kubernetes: {deployment_name}")
                return deployment
                
            except Exception as e:
                deployment.status = 'failed'
                deployment.end_time = datetime.now().isoformat()
                deployment.logs.append(f"Deployment failed: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to deploy to Kubernetes: {str(e)}")
            raise

    async def deploy_aws(
        self,
        bucket: str,
        source_path: Union[str, Path],
        prefix: str = '',
        acl: str = 'private'
    ) -> Deployment:
        """Deploy documentation to AWS S3."""
        try:
            # Generate deployment ID
            deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment record
            deployment = Deployment(
                id=deployment_id,
                type='aws',
                status='in_progress',
                start_time=datetime.now().isoformat(),
                end_time=None,
                config={
                    'bucket': bucket,
                    'source_path': str(source_path),
                    'prefix': prefix,
                    'acl': acl
                },
                logs=[],
                artifacts=[]
            )
            
            self.deployments[deployment_id] = deployment
            
            try:
                # Run AWS operations in a thread pool
                loop = asyncio.get_event_loop()
                
                # Upload files
                source_path = Path(source_path)
                if source_path.is_file():
                    # Upload single file
                    key = f"{prefix}/{source_path.name}" if prefix else source_path.name
                    await loop.run_in_executor(
                        None,
                        lambda: self.aws_client.upload_file(
                            str(source_path),
                            bucket,
                            key,
                            ExtraArgs={'ACL': acl}
                        )
                    )
                    deployment.artifacts.append(key)
                else:
                    # Upload directory
                    for file_path in source_path.rglob("*"):
                        if file_path.is_file():
                            key = f"{prefix}/{file_path.relative_to(source_path)}" if prefix else str(file_path.relative_to(source_path))
                            await loop.run_in_executor(
                                None,
                                lambda: self.aws_client.upload_file(
                                    str(file_path),
                                    bucket,
                                    key,
                                    ExtraArgs={'ACL': acl}
                                )
                            )
                            deployment.artifacts.append(key)
                
                # Update deployment
                deployment.status = 'completed'
                deployment.end_time = datetime.now().isoformat()
                
                self.logger.info(f"Deployed to AWS S3: {bucket}")
                return deployment
                
            except Exception as e:
                deployment.status = 'failed'
                deployment.end_time = datetime.now().isoformat()
                deployment.logs.append(f"Deployment failed: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to deploy to AWS S3: {str(e)}")
            raise

    async def deploy_gcs(
        self,
        bucket: str,
        source_path: Union[str, Path],
        prefix: str = '',
        public: bool = False
    ) -> Deployment:
        """Deploy documentation to Google Cloud Storage."""
        try:
            # Generate deployment ID
            deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment record
            deployment = Deployment(
                id=deployment_id,
                type='gcs',
                status='in_progress',
                start_time=datetime.now().isoformat(),
                end_time=None,
                config={
                    'bucket': bucket,
                    'source_path': str(source_path),
                    'prefix': prefix,
                    'public': public
                },
                logs=[],
                artifacts=[]
            )
            
            self.deployments[deployment_id] = deployment
            
            try:
                # Run GCS operations in a thread pool
                loop = asyncio.get_event_loop()
                
                # Get bucket
                bucket_obj = await loop.run_in_executor(
                    None,
                    lambda: self.gcs_client.bucket(bucket)
                )
                
                # Upload files
                source_path = Path(source_path)
                if source_path.is_file():
                    # Upload single file
                    key = f"{prefix}/{source_path.name}" if prefix else source_path.name
                    blob = bucket_obj.blob(key)
                    await loop.run_in_executor(
                        None,
                        lambda: blob.upload_from_filename(str(source_path))
                    )
                    if public:
                        await loop.run_in_executor(
                            None,
                            lambda: blob.make_public()
                        )
                    deployment.artifacts.append(key)
                else:
                    # Upload directory
                    for file_path in source_path.rglob("*"):
                        if file_path.is_file():
                            key = f"{prefix}/{file_path.relative_to(source_path)}" if prefix else str(file_path.relative_to(source_path))
                            blob = bucket_obj.blob(key)
                            await loop.run_in_executor(
                                None,
                                lambda: blob.upload_from_filename(str(file_path))
                            )
                            if public:
                                await loop.run_in_executor(
                                    None,
                                    lambda: blob.make_public()
                                )
                            deployment.artifacts.append(key)
                
                # Update deployment
                deployment.status = 'completed'
                deployment.end_time = datetime.now().isoformat()
                
                self.logger.info(f"Deployed to GCS: {bucket}")
                return deployment
                
            except Exception as e:
                deployment.status = 'failed'
                deployment.end_time = datetime.now().isoformat()
                deployment.logs.append(f"Deployment failed: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to deploy to GCS: {str(e)}")
            raise

    async def deploy_azure(
        self,
        container: str,
        source_path: Union[str, Path],
        prefix: str = '',
        public: bool = False
    ) -> Deployment:
        """Deploy documentation to Azure Blob Storage."""
        try:
            # Generate deployment ID
            deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment record
            deployment = Deployment(
                id=deployment_id,
                type='azure',
                status='in_progress',
                start_time=datetime.now().isoformat(),
                end_time=None,
                config={
                    'container': container,
                    'source_path': str(source_path),
                    'prefix': prefix,
                    'public': public
                },
                logs=[],
                artifacts=[]
            )
            
            self.deployments[deployment_id] = deployment
            
            try:
                # Run Azure operations in a thread pool
                loop = asyncio.get_event_loop()
                
                # Get container client
                container_client = await loop.run_in_executor(
                    None,
                    lambda: self.azure_client.get_container_client(container)
                )
                
                # Upload files
                source_path = Path(source_path)
                if source_path.is_file():
                    # Upload single file
                    key = f"{prefix}/{source_path.name}" if prefix else source_path.name
                    with open(source_path, 'rb') as f:
                        await loop.run_in_executor(
                            None,
                            lambda: container_client.upload_blob(
                                name=key,
                                data=f,
                                overwrite=True
                            )
                        )
                    if public:
                        await loop.run_in_executor(
                            None,
                            lambda: container_client.set_blob_access_tier(
                                key,
                                access_tier='Hot'
                            )
                        )
                    deployment.artifacts.append(key)
                else:
                    # Upload directory
                    for file_path in source_path.rglob("*"):
                        if file_path.is_file():
                            key = f"{prefix}/{file_path.relative_to(source_path)}" if prefix else str(file_path.relative_to(source_path))
                            with open(file_path, 'rb') as f:
                                await loop.run_in_executor(
                                    None,
                                    lambda: container_client.upload_blob(
                                        name=key,
                                        data=f,
                                        overwrite=True
                                    )
                                )
                            if public:
                                await loop.run_in_executor(
                                    None,
                                    lambda: container_client.set_blob_access_tier(
                                        key,
                                        access_tier='Hot'
                                    )
                                )
                            deployment.artifacts.append(key)
                
                # Update deployment
                deployment.status = 'completed'
                deployment.end_time = datetime.now().isoformat()
                
                self.logger.info(f"Deployed to Azure: {container}")
                return deployment
                
            except Exception as e:
                deployment.status = 'failed'
                deployment.end_time = datetime.now().isoformat()
                deployment.logs.append(f"Deployment failed: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to deploy to Azure: {str(e)}")
            raise

    async def deploy_ssh(
        self,
        host: str,
        username: str,
        password: Optional[str] = None,
        key_filename: Optional[str] = None,
        source_path: Union[str, Path] = None,
        target_path: Union[str, Path] = None,
        commands: Optional[List[str]] = None
    ) -> Deployment:
        """Deploy documentation using SSH."""
        try:
            # Generate deployment ID
            deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment record
            deployment = Deployment(
                id=deployment_id,
                type='ssh',
                status='in_progress',
                start_time=datetime.now().isoformat(),
                end_time=None,
                config={
                    'host': host,
                    'username': username,
                    'source_path': str(source_path) if source_path else None,
                    'target_path': str(target_path) if target_path else None,
                    'commands': commands
                },
                logs=[],
                artifacts=[]
            )
            
            self.deployments[deployment_id] = deployment
            
            try:
                # Run SSH operations in a thread pool
                loop = asyncio.get_event_loop()
                
                # Create SSH client
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                # Connect
                await loop.run_in_executor(
                    None,
                    lambda: ssh.connect(
                        hostname=host,
                        username=username,
                        password=password,
                        key_filename=key_filename
                    )
                )
                
                # Execute commands
                if commands:
                    for cmd in commands:
                        stdin, stdout, stderr = await loop.run_in_executor(
                            None,
                            lambda: ssh.exec_command(cmd)
                        )
                        output = await loop.run_in_executor(None, stdout.read)
                        error = await loop.run_in_executor(None, stderr.read)
                        deployment.logs.append(f"Command: {cmd}")
                        if output:
                            deployment.logs.append(f"Output: {output.decode()}")
                        if error:
                            deployment.logs.append(f"Error: {error.decode()}")
                
                # Transfer files
                if source_path and target_path:
                    sftp = await loop.run_in_executor(None, ssh.open_sftp)
                    source_path = Path(source_path)
                    if source_path.is_file():
                        # Upload single file
                        await loop.run_in_executor(
                            None,
                            lambda: sftp.put(
                                str(source_path),
                                str(target_path)
                            )
                        )
                        deployment.artifacts.append(str(target_path))
                    else:
                        # Upload directory
                        for file_path in source_path.rglob("*"):
                            if file_path.is_file():
                                target_file = target_path / file_path.relative_to(source_path)
                                target_file.parent.mkdir(parents=True, exist_ok=True)
                                await loop.run_in_executor(
                                    None,
                                    lambda: sftp.put(
                                        str(file_path),
                                        str(target_file)
                                    )
                                )
                                deployment.artifacts.append(str(target_file))
                
                # Close connection
                await loop.run_in_executor(None, ssh.close)
                
                # Update deployment
                deployment.status = 'completed'
                deployment.end_time = datetime.now().isoformat()
                
                self.logger.info(f"Deployed via SSH: {host}")
                return deployment
                
            except Exception as e:
                deployment.status = 'failed'
                deployment.end_time = datetime.now().isoformat()
                deployment.logs.append(f"Deployment failed: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to deploy via SSH: {str(e)}")
            raise

    async def deploy_ansible(
        self,
        inventory_path: Union[str, Path],
        playbook_path: Union[str, Path],
        extra_vars: Optional[Dict] = None
    ) -> Deployment:
        """Deploy documentation using Ansible."""
        try:
            # Generate deployment ID
            deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create deployment record
            deployment = Deployment(
                id=deployment_id,
                type='ansible',
                status='in_progress',
                start_time=datetime.now().isoformat(),
                end_time=None,
                config={
                    'inventory_path': str(inventory_path),
                    'playbook_path': str(playbook_path),
                    'extra_vars': extra_vars
                },
                logs=[],
                artifacts=[]
            )
            
            self.deployments[deployment_id] = deployment
            
            try:
                # Run Ansible operations in a thread pool
                loop = asyncio.get_event_loop()
                
                # Setup Ansible
                loader = ansible.parsing.dataloader.DataLoader()
                inventory = ansible.inventory.manager.InventoryManager(
                    loader=loader,
                    sources=str(inventory_path)
                )
                variable_manager = ansible.vars.manager.VariableManager(
                    loader=loader,
                    inventory=inventory
                )
                
                # Add extra vars
                if extra_vars:
                    variable_manager.extra_vars = extra_vars
                
                # Create callback
                class ResultCallback(ansible.plugins.callback.CallbackBase):
                    def v2_runner_on_ok(self, result, **kwargs):
                        deployment.logs.append(f"Task {result.task_name} completed")
                        
                    def v2_runner_on_failed(self, result, **kwargs):
                        deployment.logs.append(f"Task {result.task_name} failed: {result._result.get('msg', '')}")
                
                callback = ResultCallback()
                
                # Create playbook executor
                pbex = ansible.executor.playbook_executor.PlaybookExecutor(
                    playbooks=[str(playbook_path)],
                    inventory=inventory,
                    variable_manager=variable_manager,
                    loader=loader,
                    passwords={}
                )
                pbex._tqm._stdout_callback = callback
                
                # Run playbook
                await loop.run_in_executor(None, pbex.run)
                
                # Update deployment
                deployment.status = 'completed'
                deployment.end_time = datetime.now().isoformat()
                
                self.logger.info(f"Deployed via Ansible: {playbook_path}")
                return deployment
                
            except Exception as e:
                deployment.status = 'failed'
                deployment.end_time = datetime.now().isoformat()
                deployment.logs.append(f"Deployment failed: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to deploy via Ansible: {str(e)}")
            raise

    def get_deployment(self, deployment_id: str) -> Optional[Deployment]:
        """Get deployment by ID."""
        return self.deployments.get(deployment_id)

    def get_deployments(
        self,
        deployment_type: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Deployment]:
        """Get deployments with optional filtering."""
        deployments = list(self.deployments.values())
        
        if deployment_type:
            deployments = [d for d in deployments if d.type == deployment_type]
        if status:
            deployments = [d for d in deployments if d.status == status]
        if start_time:
            deployments = [d for d in deployments if (
                datetime.fromisoformat(d.start_time) >= start_time
            )]
        if end_time:
            deployments = [d for d in deployments if (
                d.end_time and datetime.fromisoformat(d.end_time) <= end_time
            )]
        
        return deployments

    async def rollback_deployment(self, deployment_id: str):
        """Rollback a deployment."""
        try:
            deployment = self.get_deployment(deployment_id)
            if not deployment:
                raise ValueError(f"Deployment {deployment_id} not found")
            
            # Run rollback in a thread pool
            loop = asyncio.get_event_loop()
            
            if deployment.type == 'docker':
                # Stop and remove container
                for artifact in deployment.artifacts:
                    await loop.run_in_executor(
                        None,
                        lambda: self.docker_client.containers.get(artifact).remove(force=True)
                    )
            
            elif deployment.type == 'kubernetes':
                # Delete deployment
                for artifact in deployment.artifacts:
                    await loop.run_in_executor(
                        None,
                        lambda: self.k8s_client.delete_namespaced_deployment(
                            name=artifact,
                            namespace=deployment.config['namespace']
                        )
                    )
            
            elif deployment.type == 'aws':
                # Delete objects
                for artifact in deployment.artifacts:
                    await loop.run_in_executor(
                        None,
                        lambda: self.aws_client.delete_object(
                            Bucket=deployment.config['bucket'],
                            Key=artifact
                        )
                    )
            
            elif deployment.type == 'gcs':
                # Delete objects
                bucket = await loop.run_in_executor(
                    None,
                    lambda: self.gcs_client.bucket(deployment.config['bucket'])
                )
                for artifact in deployment.artifacts:
                    blob = bucket.blob(artifact)
                    await loop.run_in_executor(None, blob.delete)
            
            elif deployment.type == 'azure':
                # Delete blobs
                container_client = await loop.run_in_executor(
                    None,
                    lambda: self.azure_client.get_container_client(deployment.config['container'])
                )
                for artifact in deployment.artifacts:
                    await loop.run_in_executor(
                        None,
                        lambda: container_client.delete_blob(artifact)
                    )
            
            elif deployment.type == 'ssh':
                # Connect and remove files
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                await loop.run_in_executor(
                    None,
                    lambda: ssh.connect(
                        hostname=deployment.config['host'],
                        username=deployment.config['username']
                    )
                )
                
                sftp = await loop.run_in_executor(None, ssh.open_sftp)
                for artifact in deployment.artifacts:
                    await loop.run_in_executor(
                        None,
                        lambda: sftp.remove(artifact)
                    )
                
                await loop.run_in_executor(None, ssh.close)
            
            elif deployment.type == 'ansible':
                # Run cleanup playbook
                loader = ansible.parsing.dataloader.DataLoader()
                inventory = ansible.inventory.manager.InventoryManager(
                    loader=loader,
                    sources=deployment.config['inventory_path']
                )
                variable_manager = ansible.vars.manager.VariableManager(
                    loader=loader,
                    inventory=inventory
                )
                
                pbex = ansible.executor.playbook_executor.PlaybookExecutor(
                    playbooks=[f"{deployment.config['playbook_path']}.cleanup"],
                    inventory=inventory,
                    variable_manager=variable_manager,
                    loader=loader,
                    passwords={}
                )
                
                await loop.run_in_executor(None, pbex.run)
            
            # Update deployment status
            deployment.status = 'rolled_back'
            deployment.end_time = datetime.now().isoformat()
            
            self.logger.info(f"Rolled back deployment: {deployment_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to rollback deployment: {str(e)}")
            raise 