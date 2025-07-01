"""
Pipeline Runner

This module implements the pipeline runner for executing automation pipelines.

Note: This module was adapted from the legacy automation/scripts/run_pipeline.py file.
"""

import logging
import asyncio
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml
from trading.orchestrator import Orchestrator
from trading.models import Task, Workflow, TaskStatus

async def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        raise

async def setup_logging(log_path: str):
    """Configure logging."""
    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / "pipeline.log"),
            logging.StreamHandler()
        ]
    )

async def run_pipeline(config_path: str, pipeline_id: str):
    """Run automation pipeline."""
    try:
        # Load configuration
        config = await load_config(config_path)
        
        # Setup logging
        await setup_logging(config.get('log_path', 'logs/pipeline'))
        logger = logging.getLogger(__name__)
        
        # Initialize orchestrator
        orchestrator = Orchestrator(config)
        await orchestrator.start()
        
        try:
            # Execute pipeline
            await orchestrator.execute_workflow(pipeline_id)
            logger.info(f"Pipeline {pipeline_id} completed successfully")
        except Exception as e:
            logger.error(f"Error executing pipeline: {str(e)}")
            raise
        finally:
            # Stop orchestrator
            await orchestrator.stop()
    except Exception as e:
        logging.error(f"Error running pipeline: {str(e)}")
        raise

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run automation pipeline')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--pipeline', required=True, help='Pipeline ID to run')
    args = parser.parse_args()
    
    try:
        asyncio.run(run_pipeline(args.config, args.pipeline))
    except KeyboardInterrupt:
        logging.info("Pipeline execution interrupted")
    except Exception as e:
        logging.error(f"Error running pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main() 