"""Utility functions for detecting and converting file encodings to UTF-8, with agent/session observability and reporting."""

import os
import chardet
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
import json
import uuid

def setup_logger(verbose: bool = False) -> None:
    """Configure logging for the script."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def is_utf8(file_path: str) -> bool:
    """Check if a file is UTF-8 encoded.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if file is UTF-8 encoded, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False
    except Exception:
        return False

def detect_encoding(file_path: str) -> dict:
    """Detect the encoding of a file using chardet.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        dict: Dictionary containing encoding information
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        return chardet.detect(raw_data)
    except Exception as e:
        logging.error(f"Error detecting encoding for {file_path}: {e}")
        return {'encoding': None, 'confidence': 0.0}

def convert_to_utf8(file_path: str, original_encoding: str) -> (bool, Optional[str]):
    """Convert a file to UTF-8 encoding.
    
    Args:
        file_path: Path to the file to convert
        original_encoding: Original encoding of the file
        
    Returns:
        (bool, Optional[str]): (Success, Error message if any)
    """
    try:
        with open(file_path, 'r', encoding=original_encoding) as f:
            content = f.read()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, None
    except Exception as e:
        return False, str(e)

def scan_project_for_utf8(
    root_dir: str = ".",
    convert: bool = False,
    file_extensions: Optional[List[str]] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    report_path: Optional[str] = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Scan project files for non-UTF-8 encoding and optionally convert them.
    
    Args:
        root_dir: Root directory to start scanning from
        convert: Whether to automatically convert non-UTF-8 files
        file_extensions: List of file extensions to check (default: .py, .json, .csv, .txt, .md)
        agent_id: Optional agent identifier
        session_id: Optional session identifier
        report_path: Optional path to write a JSON report
        verbose: Enable verbose logging
        
    Returns:
        List[Dict]: List of results for each non-UTF-8 file
    """
    setup_logger(verbose)
    logger = logging.getLogger(__name__)
    if file_extensions is None:
        file_extensions = ['.py', '.json', '.csv', '.txt', '.md']
    if agent_id is None:
        agent_id = str(uuid.uuid4())
    if session_id is None:
        session_id = str(uuid.uuid4())
    results = []
    logger.info(f"[agent={agent_id} session={session_id}] Scanning {root_dir} for non-UTF-8 files...")
    try:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(dirpath, file)
                    
                    # Skip binary files and certain directories
                    if any(skip_dir in file_path for skip_dir in ['.git', '__pycache__', 'venv']):
                        continue
                        
                    if not is_utf8(file_path):
                        encoding_info = detect_encoding(file_path)
                        encoding = encoding_info['encoding']
                        result = {
                            'file_path': file_path,
                            'original_encoding': encoding,
                            'conversion_status': None,
                            'error': None,
                            'agent_id': agent_id,
                            'session_id': session_id
                        }
                        if encoding:
                            logger.warning(f"[agent={agent_id} session={session_id}] Non-UTF8: {file_path} ({encoding})")
                            if convert:
                                success, error = convert_to_utf8(file_path, encoding)
                                result['conversion_status'] = 'success' if success else 'failed'
                                result['error'] = error
                                if success:
                                    logger.info(f"[agent={agent_id} session={session_id}] Converted to UTF-8: {file_path}")
                                else:
                                    logger.error(f"[agent={agent_id} session={session_id}] Failed to convert {file_path}: {error}")
                        else:
                            result['error'] = 'Could not detect encoding'
                            logger.error(f"[agent={agent_id} session={session_id}] Could not detect encoding for {file_path}")
                        results.append(result)
    except Exception as e:
        logger.error(f"[agent={agent_id} session={session_id}] Error during scanning: {e}")
        
    if not results:
        logger.info(f"[agent={agent_id} session={session_id}] All scanned files are UTF-8 encoded.")
    else:
        logger.info(f"[agent={agent_id} session={session_id}] Found {len(results)} non-UTF-8 files.")
        if not convert:
            logger.info(f"[agent={agent_id} session={session_id}] Run with convert=True to automatically convert files to UTF-8.")
            
    if report_path:
        try:
            Path(report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"[agent={agent_id} session={session_id}] Report written to {report_path}")
        except Exception as e:
            logger.error(f"[agent={agent_id} session={session_id}] Failed to write report: {e}")
            
    return results

def main():
    """Main function to run the UTF-8 scanner with CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scan and convert files to UTF-8 encoding with agent observability.")
    parser.add_argument("--root_dir", default=".", help="Root directory to scan")
    parser.add_argument("--convert", action="store_true", help="Convert non-UTF-8 files")
    parser.add_argument("--extensions", nargs="+", help="File extensions to check (default: .py .json .csv .txt .md)")
    parser.add_argument("--agent_id", default=None, help="Agent ID for logging and reporting")
    parser.add_argument("--session_id", default=None, help="Session ID for logging and reporting")
    parser.add_argument("--report_path", default=None, help="Path to write a JSON report of results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    results = scan_project_for_utf8(
        root_dir=args.root_dir,
        convert=args.convert,
        file_extensions=args.extensions,
        agent_id=args.agent_id,
        session_id=args.session_id,
        report_path=args.report_path,
        verbose=args.verbose
    )
    
    if results:
        print("\nNon-UTF-8 files found:")
        for r in results:
            print(f"- {r['file_path']} ({r['original_encoding']}) [status: {r['conversion_status'] or 'not converted'}] [error: {r['error'] or 'none'}]")
    else:
        print("\nAll scanned files are UTF-8 encoded.")

if __name__ == "__main__":
    main() 