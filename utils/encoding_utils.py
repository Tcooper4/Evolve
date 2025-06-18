"""
Encoding Utilities.

This module provides utilities for handling file encodings, including:
- Encoding detection
- Encoding conversion
- Non-UTF-8 file logging
- Encoding validation
"""

import chardet
import codecs
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class EncodingUtils:
    def __init__(self, log_dir: str = "logs"):
        """Initialize encoding utilities."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.encoding_log = self.log_dir / "encoding_report.json"
        
    def detect_encoding(self, filepath: str) -> Tuple[str, float]:
        """Detect file encoding and confidence."""
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                return result['encoding'], result['confidence']
        except Exception as e:
            self.logger.error(f"Error detecting encoding for {filepath}: {str(e)}")
            return 'unknown', 0.0
            
    def convert_encoding(
        self,
        filepath: str,
        target_encoding: str = 'utf-8',
        source_encoding: Optional[str] = None
    ) -> bool:
        """Convert file encoding."""
        try:
            # Detect source encoding if not provided
            if not source_encoding:
                source_encoding, _ = self.detect_encoding(filepath)
                
            # Skip if already in target encoding
            if source_encoding.lower() == target_encoding.lower():
                return True
                
            # Read file with source encoding
            with codecs.open(filepath, 'r', encoding=source_encoding) as f:
                content = f.read()
                
            # Write file with target encoding
            with codecs.open(filepath, 'w', encoding=target_encoding) as f:
                f.write(content)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting encoding for {filepath}: {str(e)}")
            return False
            
    def validate_encoding(self, filepath: str, encoding: str = 'utf-8') -> bool:
        """Validate file encoding."""
        try:
            with codecs.open(filepath, 'r', encoding=encoding) as f:
                f.read()
            return True
        except UnicodeDecodeError:
            return False
        except Exception as e:
            self.logger.error(f"Error validating encoding for {filepath}: {str(e)}")
            return False
            
    def log_non_utf8_file(
        self,
        filepath: str,
        encoding: str,
        confidence: float,
        error: Optional[str] = None
    ) -> None:
        """Log non-UTF-8 file information."""
        try:
            # Create log entry
            entry = {
                "filepath": filepath,
                "encoding": encoding,
                "confidence": confidence,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "file_size": os.path.getsize(filepath),
                "file_hash": self._calculate_file_hash(filepath)
            }
            
            # Load existing log
            if self.encoding_log.exists():
                with open(self.encoding_log, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = []
                
            # Add new entry
            log_data.append(entry)
            
            # Save updated log
            with open(self.encoding_log, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging non-UTF-8 file: {str(e)}")
            
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate file hash."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {str(e)}")
            return "unknown"
            
    def scan_directory(
        self,
        directory: str,
        target_encoding: str = 'utf-8',
        convert: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Scan directory for encoding issues."""
        try:
            results = {
                "non_utf8": [],
                "converted": [],
                "failed": []
            }
            
            for root, _, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # Skip binary files
                    if self._is_binary_file(filepath):
                        continue
                        
                    # Detect encoding
                    encoding, confidence = self.detect_encoding(filepath)
                    
                    # Check if non-UTF-8
                    if encoding.lower() != target_encoding.lower():
                        # Log non-UTF-8 file
                        self.log_non_utf8_file(filepath, encoding, confidence)
                        
                        # Convert if requested
                        if convert:
                            if self.convert_encoding(filepath, target_encoding, encoding):
                                results["converted"].append({
                                    "filepath": filepath,
                                    "old_encoding": encoding,
                                    "new_encoding": target_encoding
                                })
                            else:
                                results["failed"].append({
                                    "filepath": filepath,
                                    "encoding": encoding,
                                    "error": "Conversion failed"
                                })
                        else:
                            results["non_utf8"].append({
                                "filepath": filepath,
                                "encoding": encoding,
                                "confidence": confidence
                            })
                            
            return results
            
        except Exception as e:
            self.logger.error(f"Error scanning directory: {str(e)}")
            return {
                "non_utf8": [],
                "converted": [],
                "failed": []
            }
            
    def _is_binary_file(self, filepath: str) -> bool:
        """Check if file is binary."""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk
        except Exception as e:
            self.logger.error(f"Error checking binary file: {str(e)}")
            return False
            
    def generate_encoding_report(self, directory: str) -> None:
        """Generate encoding report for directory."""
        try:
            # Scan directory
            results = self.scan_directory(directory)
            
            # Create report
            report = {
                "timestamp": datetime.now().isoformat(),
                "directory": directory,
                "total_files": len(results["non_utf8"]) + len(results["converted"]) + len(results["failed"]),
                "non_utf8_files": len(results["non_utf8"]),
                "converted_files": len(results["converted"]),
                "failed_files": len(results["failed"]),
                "details": results
            }
            
            # Save report
            report_path = self.log_dir / "encoding_scan_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error generating encoding report: {str(e)}")
            
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create encoding utils
    utils = EncodingUtils()
    
    # Generate report for current directory
    utils.generate_encoding_report(".") 