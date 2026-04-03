"""
Config file watcher for hot reload

Watches configuration files and automatically reloads when changes are detected.
"""

import os
import logging
from typing import Callable, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("watchdog not available - config hot-reload disabled")


class ConfigReloader(FileSystemEventHandler):
    """Watches config files and reloads on change"""
    
    def __init__(self, config_file: str, reload_callback: Callable):
        """
        Initialize config reloader.
        
        Args:
            config_file: Path to config file to watch
            reload_callback: Callback function to call when file changes
        """
        self.config_file = Path(config_file).resolve()
        self.reload_callback = reload_callback
        logger.info(f"Watching config file: {self.config_file}")
    
    def on_modified(self, event):
        """Handle file modification event"""
        if not event.is_directory:
            event_path = Path(event.src_path).resolve()
            if event_path == self.config_file:
                logger.info(f"Config file changed: {self.config_file}")
                try:
                    self.reload_callback()
                    logger.info("Config reloaded successfully")
                except Exception as e:
                    logger.error(f"Error reloading config: {e}")


class ConfigWatcher:
    """Manages config file watching"""
    
    def __init__(self):
        self.observers = []
        self.watched_files = []
    
    def watch_config(
        self,
        config_file: str,
        reload_callback: Callable,
        recursive: bool = False
    ) -> Optional[Observer]:
        """
        Start watching a config file.
        
        Args:
            config_file: Path to config file
            reload_callback: Callback to call on file change
            recursive: Whether to watch directory recursively
            
        Returns:
            Observer instance or None if watchdog not available
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available - cannot watch config files")
            return None
        
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Config file does not exist: {config_file}")
                return None
            
            event_handler = ConfigReloader(config_file, reload_callback)
            observer = Observer()
            
            watch_path = config_path.parent if not recursive else config_path
            observer.schedule(event_handler, str(watch_path), recursive=recursive)
            observer.start()
            
            self.observers.append(observer)
            self.watched_files.append(config_file)
            
            logger.info(f"Started watching config file: {config_file}")
            return observer
            
        except Exception as e:
            logger.error(f"Error setting up config watcher: {e}")
            return None
    
    def stop_all(self):
        """Stop watching all config files"""
        for observer in self.observers:
            try:
                observer.stop()
                observer.join()
            except Exception as e:
                logger.error(f"Error stopping observer: {e}")
        
        self.observers.clear()
        self.watched_files.clear()
        logger.info("Stopped all config watchers")


# Global config watcher instance
_config_watcher = ConfigWatcher()


def watch_config(
    config_file: str,
    reload_callback: Callable,
    recursive: bool = False
) -> Optional[Observer]:
    """
    Start watching a config file for changes.
    
    Args:
        config_file: Path to config file
        reload_callback: Callback to call when file changes
        recursive: Whether to watch directory recursively
        
    Returns:
        Observer instance or None if watchdog not available
    
    Example:
        def reload_config():
            config.reload()
        
        watcher = watch_config('config.yaml', reload_config)
    """
    return _config_watcher.watch_config(config_file, reload_callback, recursive)


def stop_config_watchers():
    """Stop all config watchers"""
    _config_watcher.stop_all()

