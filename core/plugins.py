"""
Plugin Architecture System

This module provides a plugin system for dynamic loading and management
of system components, enabling extensibility and modularity.
"""

import os
import sys
import importlib
import importlib.util
import logging
from typing import Any, Dict, List, Optional, Type, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import yaml
from datetime import datetime

from .interfaces import IPlugin, IEventBus, EventType, SystemEvent
from .container import ServiceContainer

logger = logging.getLogger(__name__)

class PluginStatus(Enum):
    """Plugin status enumeration."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class PluginInfo:
    """Plugin information and metadata."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    capabilities: List[str]
    config_schema: Optional[Dict[str, Any]] = None
    entry_point: Optional[str] = None
    file_path: Optional[str] = None
    status: PluginStatus = PluginStatus.DISCOVERED
    error_message: Optional[str] = None
    load_time: Optional[datetime] = None
    init_time: Optional[datetime] = None

class PluginManager:
    """
    Plugin manager for dynamic loading and management of system plugins.
    
    Features:
    - Automatic plugin discovery
    - Dependency resolution
    - Configuration management
    - Hot reloading
    - Plugin lifecycle management
    - Service integration
    """
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None, container: Optional[ServiceContainer] = None):
        """
        Initialize the plugin manager.
        
        Args:
            plugin_dirs: Directories to search for plugins
            container: Service container for dependency injection
        """
        self.plugin_dirs = plugin_dirs or ["./plugins", "./trading/plugins"]
        self.container = container
        self.plugins: Dict[str, PluginInfo] = {}
        self.loaded_plugins: Dict[str, IPlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.disabled_plugins: Set[str] = set()
        
        # Load plugin configurations
        self._load_plugin_configs()
        
        # Discover plugins
        self.discover_plugins()
    
    def _load_plugin_configs(self) -> None:
        """Load plugin configurations from files."""
        config_files = [
            "plugin_config.json",
            "plugin_config.yaml",
            "plugin_config.yml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        if config_file.endswith('.json'):
                            config = json.load(f)
                        else:
                            config = yaml.safe_load(f)
                    
                    if 'plugins' in config:
                        for plugin_name, plugin_config in config['plugins'].items():
                            self.plugin_configs[plugin_name] = plugin_config
                    
                    if 'disabled_plugins' in config:
                        self.disabled_plugins.update(config['disabled_plugins'])
                    
                    logger.info(f"Loaded plugin configuration from {config_file}")
                except Exception as e:
                    logger.error(f"Error loading plugin config {config_file}: {e}")
    
    def discover_plugins(self) -> List[PluginInfo]:
        """
        Discover plugins in the configured directories.
        
        Returns:
            List of discovered plugin information
        """
        discovered_plugins = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
            
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                
                if os.path.isdir(item_path):
                    # Directory-based plugin
                    plugin_info = self._discover_directory_plugin(item_path, item)
                elif item.endswith('.py') and not item.startswith('__'):
                    # Single file plugin
                    plugin_info = self._discover_file_plugin(item_path, item[:-3])
                else:
                    continue
                
                if plugin_info:
                    discovered_plugins.append(plugin_info)
                    self.plugins[plugin_info.name] = plugin_info
        
        logger.info(f"Discovered {len(discovered_plugins)} plugins")
        return discovered_plugins
    
    def _discover_directory_plugin(self, plugin_path: str, plugin_name: str) -> Optional[PluginInfo]:
        """Discover a directory-based plugin."""
        # Look for plugin manifest
        manifest_files = ["plugin.json", "plugin.yaml", "plugin.yml", "__init__.py"]
        
        for manifest_file in manifest_files:
            manifest_path = os.path.join(plugin_path, manifest_file)
            if os.path.exists(manifest_path):
                return self._parse_plugin_manifest(manifest_path, plugin_name, plugin_path)
        
        # Try to infer plugin info from directory structure
        return self._infer_plugin_info(plugin_path, plugin_name)
    
    def _discover_file_plugin(self, plugin_path: str, plugin_name: str) -> Optional[PluginInfo]:
        """Discover a single file plugin."""
        try:
            # Try to import the module to get metadata
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for plugin metadata
                if hasattr(module, 'PLUGIN_INFO'):
                    info = module.PLUGIN_INFO
                    return PluginInfo(
                        name=info.get('name', plugin_name),
                        version=info.get('version', '1.0.0'),
                        description=info.get('description', ''),
                        author=info.get('author', 'Unknown'),
                        dependencies=info.get('dependencies', []),
                        capabilities=info.get('capabilities', []),
                        config_schema=info.get('config_schema'),
                        entry_point=info.get('entry_point'),
                        file_path=plugin_path
                    )
                
                # Try to find plugin class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, IPlugin) and attr != IPlugin:
                        return PluginInfo(
                            name=plugin_name,
                            version='1.0.0',
                            description=f'Plugin from {plugin_name}',
                            author='Unknown',
                            dependencies=[],
                            capabilities=[],
                            entry_point=f'{plugin_name}.{attr_name}',
                            file_path=plugin_path
                        )
        
        except Exception as e:
            logger.error(f"Error discovering file plugin {plugin_path}: {e}")
        
        return None
    
    def _parse_plugin_manifest(self, manifest_path: str, plugin_name: str, plugin_path: str) -> Optional[PluginInfo]:
        """Parse a plugin manifest file."""
        try:
            with open(manifest_path, 'r') as f:
                if manifest_path.endswith('.json'):
                    manifest = json.load(f)
                else:
                    manifest = yaml.safe_load(f)
            
            return PluginInfo(
                name=manifest.get('name', plugin_name),
                version=manifest.get('version', '1.0.0'),
                description=manifest.get('description', ''),
                author=manifest.get('author', 'Unknown'),
                dependencies=manifest.get('dependencies', []),
                capabilities=manifest.get('capabilities', []),
                config_schema=manifest.get('config_schema'),
                entry_point=manifest.get('entry_point'),
                file_path=plugin_path
            )
        
        except Exception as e:
            logger.error(f"Error parsing plugin manifest {manifest_path}: {e}")
            return None
    
    def _infer_plugin_info(self, plugin_path: str, plugin_name: str) -> Optional[PluginInfo]:
        """Infer plugin information from directory structure."""
        # Look for common files
        has_init = os.path.exists(os.path.join(plugin_path, '__init__.py'))
        has_readme = any(f.endswith('README') for f in os.listdir(plugin_path))
        
        if has_init:
            return PluginInfo(
                name=plugin_name,
                version='1.0.0',
                description=f'Plugin from {plugin_name}',
                author='Unknown',
                dependencies=[],
                capabilities=[],
                file_path=plugin_path
            )
        
        return None
    
    def load_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """
        Load a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            Loaded plugin instance or None if failed
        """
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} not found")
            return None
        
        if plugin_name in self.disabled_plugins:
            logger.warning(f"Plugin {plugin_name} is disabled")
            return None
        
        plugin_info = self.plugins[plugin_name]
        
        try:
            # Check dependencies
            if not self._check_dependencies(plugin_info.dependencies):
                logger.error(f"Plugin {plugin_name} dependencies not met")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Dependencies not met"
                return None
            
            # Load the plugin
            plugin_instance = self._load_plugin_instance(plugin_info)
            if not plugin_instance:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Failed to load plugin instance"
                return None
            
            # Initialize the plugin
            config = self.plugin_configs.get(plugin_name, {})
            if not plugin_instance.initialize(config):
                logger.error(f"Failed to initialize plugin {plugin_name}")
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_message = "Initialization failed"
                return None
            
            # Register with service container if available
            if self.container:
                self._register_plugin_services(plugin_instance, plugin_name)
            
            # Update status
            plugin_info.status = PluginStatus.INITIALIZED
            plugin_info.init_time = datetime.now()
            self.loaded_plugins[plugin_name] = plugin_instance
            
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return plugin_instance
        
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_message = str(e)
            return None
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if plugin dependencies are available."""
        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                logger.warning(f"Dependency not available: {dep}")
                return False
        return True
    
    def _load_plugin_instance(self, plugin_info: PluginInfo) -> Optional[IPlugin]:
        """Load a plugin instance."""
        try:
            if plugin_info.entry_point:
                # Load from entry point
                module_name, class_name = plugin_info.entry_point.rsplit('.', 1)
                module = importlib.import_module(module_name)
                plugin_class = getattr(module, class_name)
            elif plugin_info.file_path:
                # Load from file path
                if plugin_info.file_path.endswith('.py'):
                    spec = importlib.util.spec_from_file_location(
                        plugin_info.name, plugin_info.file_path
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find plugin class
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, IPlugin) and attr != IPlugin:
                            plugin_class = attr
                            break
                    else:
                        logger.error(f"No plugin class found in {plugin_info.file_path}")
                        return None
                else:
                    # Directory-based plugin
                    module = importlib.import_module(plugin_info.name)
                    plugin_class = getattr(module, 'Plugin', None)
                    if not plugin_class:
                        logger.error(f"No Plugin class found in {plugin_info.name}")
                        return None
            else:
                logger.error(f"No entry point or file path for plugin {plugin_info.name}")
                return None
            
            # Create instance
            plugin_instance = plugin_class()
            plugin_info.status = PluginStatus.LOADED
            plugin_info.load_time = datetime.now()
            
            return plugin_instance
        
        except Exception as e:
            logger.error(f"Error loading plugin instance for {plugin_info.name}: {e}")
            return None
    
    def _register_plugin_services(self, plugin: IPlugin, plugin_name: str) -> None:
        """Register plugin services with the service container."""
        try:
            # Register the plugin itself
            self.container.register_singleton(IPlugin, plugin)
            
            # Register plugin capabilities as services
            capabilities = plugin.get_capabilities()
            for capability in capabilities:
                # Try to register capability as a service
                # This is a simplified approach - in practice you'd want more sophisticated service registration
                logger.debug(f"Registering plugin capability: {capability}")
        
        except Exception as e:
            logger.error(f"Error registering plugin services for {plugin_name}: {e}")
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if successfully unloaded
        """
        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return False
        
        try:
            plugin = self.loaded_plugins[plugin_name]
            plugin.cleanup()
            
            del self.loaded_plugins[plugin_name]
            self.plugins[plugin_name].status = PluginStatus.DISCOVERED
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Get a loaded plugin by name."""
        return self.loaded_plugins.get(plugin_name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information by name."""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self, status: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """List all plugins, optionally filtered by status."""
        if status:
            return [p for p in self.plugins.values() if p.status == status]
        return list(self.plugins.values())
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a disabled plugin."""
        if plugin_name in self.disabled_plugins:
            self.disabled_plugins.remove(plugin_name)
            logger.info(f"Enabled plugin: {plugin_name}")
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.loaded_plugins:
            self.unload_plugin(plugin_name)
        
        self.disabled_plugins.add(plugin_name)
        logger.info(f"Disabled plugin: {plugin_name}")
        return True
    
    def reload_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Reload a plugin."""
        if plugin_name in self.loaded_plugins:
            self.unload_plugin(plugin_name)
        
        return self.load_plugin(plugin_name)
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        return {
            'total_plugins': len(self.plugins),
            'loaded_plugins': len(self.loaded_plugins),
            'disabled_plugins': len(self.disabled_plugins),
            'plugins_by_status': {
                status.value: len([p for p in self.plugins.values() if p.status == status])
                for status in PluginStatus
            }
        }

# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

def set_plugin_manager(plugin_manager: PluginManager) -> None:
    """Set the global plugin manager instance."""
    global _plugin_manager
    _plugin_manager = plugin_manager

__all__ = [
    'PluginManager', 'PluginInfo', 'PluginStatus',
    'get_plugin_manager', 'set_plugin_manager'
] 