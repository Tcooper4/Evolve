"""System status monitoring utility.

This module provides functions to check the health of various system components
and return an overall system status.
"""

import logging
from typing import Dict, Any
from datetime import datetime
import psutil
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def check_disk_space() -> Dict[str, Any]:
    """Check available disk space.
    
    Returns:
        Dictionary with disk space status
    """
    try:
        disk = psutil.disk_usage('/')
        return {
            'status': 'operational' if disk.percent < 90 else 'degraded',
            'percent_used': disk.percent,
            'free_gb': disk.free / (1024**3)
        }
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        return {'status': 'down', 'error': str(e)}

def check_memory_usage() -> Dict[str, Any]:
    """Check memory usage.
    
    Returns:
        Dictionary with memory status
    """
    try:
        memory = psutil.virtual_memory()
        return {
            'status': 'operational' if memory.percent < 90 else 'degraded',
            'percent_used': memory.percent,
            'free_gb': memory.available / (1024**3)
        }
    except Exception as e:
        logger.error(f"Error checking memory: {str(e)}")
        return {'status': 'down', 'error': str(e)}

def check_model_health() -> Dict[str, Any]:
    """Check health of model files and directories.
    
    Returns:
        Dictionary with model health status
    """
    try:
        model_dir = Path('models')
        if not model_dir.exists():
            return {'status': 'down', 'error': 'Model directory not found'}
            
        # Check if models directory has any Python files (more flexible than requiring specific files)
        model_files = list(model_dir.glob('*.py'))
        if not model_files:
            return {
                'status': 'degraded',
                'message': 'No Python model files found'
            }
            
        return {'status': 'operational'}
        
    except Exception as e:
        logger.error(f"Error checking model health: {str(e)}")
        return {'status': 'down', 'error': str(e)}

def check_data_health() -> Dict[str, Any]:
    """Check health of data files and directories.
    
    Returns:
        Dictionary with data health status
    """
    try:
        data_dir = Path('data')
        if not data_dir.exists():
            return {'status': 'down', 'error': 'Data directory not found'}
            
        # Check if data directory has any files (more flexible than requiring specific files)
        data_files = list(data_dir.glob('*'))
        if not data_files:
            return {
                'status': 'degraded',
                'message': 'Data directory is empty'
            }
            
        return {'status': 'operational'}
        
    except Exception as e:
        logger.error(f"Error checking data health: {str(e)}")
        return {'status': 'down', 'error': str(e)}

def get_system_status() -> Dict[str, Any]:
    """Get overall system status.
    
    Returns:
        Dictionary with system status and component details
    """
    try:
        # Check individual components
        disk_status = check_disk_space()
        memory_status = check_memory_usage()
        model_status = check_model_health()
        data_status = check_data_health()
        
        # Determine overall status
        statuses = [
            disk_status['status'],
            memory_status['status'],
            model_status['status'],
            data_status['status']
        ]
        
        if 'down' in statuses:
            overall_status = 'down'
        elif 'degraded' in statuses:
            overall_status = 'degraded'
        else:
            overall_status = 'operational'
            
        return {
            'status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'disk': disk_status,
                'memory': memory_status,
                'models': model_status,
                'data': data_status
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {
            'status': 'down',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        } 