"""Initialize and verify logging files."""

import os
import json
from datetime import datetime
from pathlib import Path
import logging

# Required log files
REQUIRED_FILES = {
    "app.log": "[{timestamp}] System initialized\n",
    "audit.log": "[{timestamp}] Audit system initialized\n",
    "metrics.jsonl": {"timestamp": "{timestamp}", "status": "initialized"}
}

def init_log_files(log_dir: str = "logs") -> dict:
    """Initialize or verify log files.
    
    Args:
        log_dir: Directory containing log files
        
    Returns:
        Dictionary with initialization status and details
    """
    try:
        # Create logs directory if it doesn't exist
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create log_dir: {e}")
        
        # Get current timestamp
        timestamp = datetime.utcnow().isoformat()
        
        created_files = []
        verified_files = []
        errors = []
        
        # Initialize each required file
        for filename, content in REQUIRED_FILES.items():
            path = os.path.join(log_dir, filename)
            
            try:
                # Create file if it doesn't exist
                if not os.path.exists(path):
                    with open(path, "w") as file:
                        if filename.endswith(".jsonl"):
                            # JSON Lines format
                            json.dump(content.format(timestamp=timestamp), file)
                            file.write("\n")
                        else:
                            # Text format
                            file.write(content.format(timestamp=timestamp))
                    
                    created_files.append(filename)
                    logging.info(f"Created {filename}")
                else:
                    verified_files.append(filename)
                    logging.info(f"Verified {filename} exists")
                    
            except Exception as e:
                errors.append(f"Error with {filename}: {str(e)}")
                logging.error(f"Error initializing {filename}: {e}")
        
        return {
            'success': len(errors) == 0,
            'message': f'Initialized {len(created_files)} files, verified {len(verified_files)} files',
            'log_dir': log_dir,
            'created_files': created_files,
            'verified_files': verified_files,
            'errors': errors,
            'total_files': len(REQUIRED_FILES),
            'timestamp': timestamp
        }
        
    except Exception as e:
        logging.error(f"Error in init_log_files: {e}")
        return {
            'success': False,
            'message': f'Error initializing log files: {str(e)}',
            'log_dir': log_dir,
            'created_files': [],
            'verified_files': [],
            'errors': [str(e)],
            'total_files': len(REQUIRED_FILES),
            'timestamp': datetime.utcnow().isoformat()
        }

def main() -> dict:
    """Main entry point.
    
    Returns:
        Dictionary with initialization results
    """
    try:
        # Set up basic logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize log files
        result = init_log_files()
        
        if result['success']:
            logger.info("All logging files verified or recreated.")
        else:
            logger.error(f"Log initialization completed with errors: {result['errors']}")
            
        return result
        
    except Exception as e:
        error_result = {
            'success': False,
            'message': f'Error in main: {str(e)}',
            'timestamp': datetime.utcnow().isoformat()
        }
        logger.error(f"Error: {e}")
        return error_result

if __name__ == "__main__":
    main() 