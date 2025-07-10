#!/usr/bin/env python3
"""
Script to run pytest and capture all output to a log file.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_tests_with_logging():
    """Run pytest and capture all output to a log file."""
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"test_run_{timestamp}.log"
    
    print(f"Running tests and logging to: {log_file}")
    
    # Run pytest with output capture
    try:
        # Run pytest and capture all output
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "-v",
            "--tb=short",
            "--log-cli-level=INFO",
            "--log-file=test_results.log",
            "--log-file-level=INFO"
        ], 
        capture_output=True, 
        text=True,
        cwd=os.getcwd()
        )
        
        # Write all output to our log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Test Run: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write stdout
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n")
            
            # Write stderr
            if result.stderr:
                f.write("STDERR:\n")
                f.write(result.stderr)
                f.write("\n")
            
            # Write return code
            f.write(f"\nReturn Code: {result.returncode}\n")
        
        print(f"✅ Test run completed. Log saved to: {log_file}")
        print(f"Return code: {result.returncode}")
        
        # Also print summary to console
        if result.stdout:
            print("\nTest Summary:")
            print(result.stdout[-1000:])  # Last 1000 chars of output
        
        return log_file, result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return None, -1

def read_test_log(log_file):
    """Read and display the test log file."""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading log file: {e}"

if __name__ == "__main__":
    # Run tests
    log_file, return_code = run_tests_with_logging()
    
    if log_file:
        print(f"\n📄 Log file created: {log_file}")
        print("You can read the full log with: read_test_log()")
        
        # Show a preview of the log
        print("\n📋 Log Preview (last 500 chars):")
        content = read_test_log(log_file)
        print(content[-500:])
    else:
        print("❌ Failed to create log file") 