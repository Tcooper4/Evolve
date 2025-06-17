"""
# Adapted from automation/core/service_manager.py â€” legacy service management logic

Service Manager

This module implements a manager for handling service lifecycle and coordination.

Note: This module was adapted from the legacy automation/core/service_manager.py file.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
from datetime import datetime
from .base_agent import BaseAgent
import json

class ServiceManager:
    """Manager for handling service lifecycle and coordination."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the service manager."""
        self.config = config
        self.services: Dict[str, Dict[str, Any]] = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for service management."""
        log_path = Path("logs/services")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "service_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def register_service(
        self,
        name: str,
        service: BaseAgent
    ) -> None:
        """Register a new service."""
        try:
            if name in self.services:
                raise ValueError(f"Service {name} already registered")
            
            self.services[name] = {
                'config': service.config,
                'status': 'running',
                'start_time': datetime.utcnow().isoformat()
            }
            self.logger.info(f"Registered service: {name}")
        except Exception as e:
            self.logger.error(f"Error registering service {name}: {str(e)}")
            raise
    
    async def unregister_service(self, name: str) -> None:
        """Unregister a service."""
        try:
            if name in self.services:
                await self.stop_service(name)
                del self.services[name]
                self.logger.info(f"Unregistered service: {name}")
            else:
                raise ValueError(f"Unknown service: {name}")
        except Exception as e:
            self.logger.error(f"Error unregistering service {name}: {str(e)}")
            raise
    
    async def start_service(
        self,
        service_name: str,
        service_config: Dict[str, Any]
    ) -> None:
        """Start a service."""
        try:
            if service_name in self.services:
                raise ValueError(f"Service {service_name} already exists")
            
            self.services[service_name] = {
                'config': service_config,
                'status': 'running',
                'start_time': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Started service: {service_name}")
        except Exception as e:
            self.logger.error(f"Error starting service {service_name}: {str(e)}")
            raise
    
    async def stop_service(self, service_name: str) -> None:
        """Stop a service."""
        try:
            if service_name not in self.services:
                raise ValueError(f"Service {service_name} not found")
            
            del self.services[service_name]
            
            self.logger.info(f"Stopped service: {service_name}")
        except Exception as e:
            self.logger.error(f"Error stopping service {service_name}: {str(e)}")
            raise
    
    async def restart_service(self, service_name: str) -> None:
        """Restart a service."""
        try:
            if service_name not in self.services:
                raise ValueError(f"Service {service_name} not found")
            
            service_config = self.services[service_name]['config']
            await self.stop_service(service_name)
            await self.start_service(service_name, service_config)
            
            self.logger.info(f"Restarted service: {service_name}")
        except Exception as e:
            self.logger.error(f"Error restarting service {service_name}: {str(e)}")
            raise
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get service status."""
        try:
            if service_name not in self.services:
                raise ValueError(f"Service {service_name} not found")
            
            return self.services[service_name]
        except Exception as e:
            self.logger.error(f"Error getting service status: {str(e)}")
            raise
    
    def list_services(self) -> List[str]:
        """List all services."""
        return list(self.services.keys())
    
    async def monitor_services(self, interval: int = 60):
        """Monitor services at regular intervals."""
        try:
            while True:
                for service_name in self.services:
                    # TODO: Implement service health check
                    pass
                
                await asyncio.sleep(interval)
        except Exception as e:
            self.logger.error(f"Error monitoring services: {str(e)}")
            raise
    
    def update_service_config(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> None:
        """Update service configuration."""
        try:
            if service_name not in self.services:
                raise ValueError(f"Service {service_name} not found")
            
            self.services[service_name]['config'].update(config)
            self.logger.info(f"Updated service config: {service_name}")
        except Exception as e:
            self.logger.error(f"Error updating service config: {str(e)}")
            raise
    
    async def start_all_services(self) -> None:
        """Start all registered services."""
        try:
            for name in self.services:
                await self.start_service(name, self.services[name]['config'])
            self.logger.info("Started all services")
        except Exception as e:
            self.logger.error(f"Error starting all services: {str(e)}")
            raise
    
    async def stop_all_services(self) -> None:
        """Stop all registered services."""
        try:
            for name in self.services:
                await self.stop_service(name)
            self.logger.info("Stopped all services")
        except Exception as e:
            self.logger.error(f"Error stopping all services: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Start the service manager."""
        try:
            self.logger.info("Service manager started")
            
            # Start all services
            await self.start_all_services()
            
            # Start service monitoring
            asyncio.create_task(self.monitor_services())
            
        except Exception as e:
            self.logger.error(f"Error starting service manager: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the service manager."""
        try:
            # Stop all services
            await self.stop_all_services()
            
            self.logger.info("Service manager stopped")
        except Exception as e:
            self.logger.error(f"Error stopping service manager: {str(e)}")
            raise 