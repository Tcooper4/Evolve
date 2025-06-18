"""Development tool to initialize and verify logging files."""

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

def init_log_files(log_dir: str = "logs") -> None:
    """Initialize or verify log files.
    
    Args:
        log_dir: Directory containing log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.utcnow().isoformat()
    
    # Initialize each required file
    for filename, content in REQUIRED_FILES.items():
        path = os.path.join(log_dir, filename)
        
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
            
            print(f"Created {filename}")
        else:
            print(f"Verified {filename} exists")

def main() -> None:
    """Main entry point."""
    # Initialize log files
    init_log_files()
    
    print("\nAll logging files verified or recreated.")
    print("Note: These files are for development only and should not be committed to git.")

if __name__ == "__main__":
    main() 