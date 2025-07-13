"""Utility functions for detecting and converting file encodings to UTF-8."""

import logging
import os
from typing import List, Optional, Tuple

import chardet

# Configure logging
logger = logging.getLogger(__name__)


def check_encoding(file_path: str) -> dict:
    """Check the encoding of a file.

    Args:
        file_path: Path to the file to check

    Returns:
        dict: Dictionary containing encoding information
    """
    return detect_encoding(file_path)


def is_utf8(file_path: str) -> bool:
    """Check if a file is UTF-8 encoded.

    Args:
        file_path: Path to the file to check

    Returns:
        bool: True if file is UTF-8 encoded, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False


def detect_encoding(file_path: str) -> dict:
    """Detect the encoding of a file using chardet.

    Args:
        file_path: Path to the file to analyze

    Returns:
        dict: Dictionary containing encoding information
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        return chardet.detect(raw_data)
    except Exception as e:
        logger.error(f"Error detecting encoding for {file_path}: {e}")


def convert_to_utf8(file_path: str, original_encoding: str) -> bool:
    """Convert a file to UTF-8 encoding.

    Args:
        file_path: Path to the file to convert
        original_encoding: Original encoding of the file

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Read content with original encoding
        with open(file_path, "r", encoding=original_encoding) as f:
            content = f.read()

        # Write content with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Successfully converted {file_path} to UTF-8")
        return True
    except Exception as e:
        logger.error(f"Failed to convert {file_path}: {e}")
        return False


def read_with_fallback_encoding(file_path: str, preferred_encodings: Optional[List[str]] = None) -> Tuple[str, str]:
    """Read a file with robust Unicode encoding fallback when encountering malformed characters.

    Args:
        file_path: Path to the file to read
        preferred_encodings: List of encodings to try in order (default: ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1'])

    Returns:
        Tuple[str, str]: (content, used_encoding)
    """
    if preferred_encodings is None:
        preferred_encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "ascii"]

    # First try to detect encoding
    detected_info = detect_encoding(file_path)
    if detected_info and detected_info["encoding"]:
        detected_encoding = detected_info["encoding"]
        if detected_encoding not in preferred_encodings:
            preferred_encodings.insert(0, detected_encoding)

    # Try each encoding with error handling
    for encoding in preferred_encodings:
        try:
            with open(file_path, "r", encoding=encoding, errors="strict") as f:
                content = f.read()
                logger.info(f"Successfully read {file_path} with {encoding} encoding")
                return content, encoding
        except UnicodeDecodeError as e:
            logger.debug(f"Failed to read {file_path} with {encoding}: {e}")
            continue
        except Exception as e:
            logger.debug(f"Error reading {file_path} with {encoding}: {e}")
            continue

    # If all encodings fail, try with error handling
    for encoding in preferred_encodings:
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read()
                logger.warning(f"Read {file_path} with {encoding} using replacement characters")
                return content, encoding
        except Exception as e:
            logger.debug(f"Error reading {file_path} with {encoding} (replace): {e}")
            continue

    # Last resort: read as bytes and decode with replacement
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
        content = raw_data.decode("utf-8", errors="replace")
        logger.error(f"Read {file_path} as raw bytes with replacement characters")
        return content, "utf-8-replace"
    except Exception as e:
        logger.error(f"Failed to read {file_path} even with replacement: {e}")
        raise


def write_with_encoding_fallback(file_path: str, content: str, preferred_encoding: str = "utf-8") -> bool:
    """Write content to file with encoding fallback support.

    Args:
        file_path: Path to the file to write
        content: Content to write
        preferred_encoding: Preferred encoding to use

    Returns:
        bool: True if write was successful, False otherwise
    """
    try:
        # Try preferred encoding first
        with open(file_path, "w", encoding=preferred_encoding, errors="strict") as f:
            f.write(content)
        logger.info(f"Successfully wrote {file_path} with {preferred_encoding} encoding")
        return True
    except UnicodeEncodeError as e:
        logger.warning(f"Failed to write {file_path} with {preferred_encoding}: {e}")

        # Try with replacement characters
        try:
            with open(file_path, "w", encoding=preferred_encoding, errors="replace") as f:
                f.write(content)
            logger.warning(f"Wrote {file_path} with {preferred_encoding} using replacement characters")
            return True
        except Exception as e:
            logger.error(f"Failed to write {file_path} with {preferred_encoding} (replace): {e}")

            # Try UTF-8 as fallback
            try:
                with open(file_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(content)
                logger.warning(f"Wrote {file_path} with UTF-8 using replacement characters")
                return True
            except Exception as e:
                logger.error(f"Failed to write {file_path} with UTF-8: {e}")
                return False


def sanitize_unicode_content(content: str, remove_control_chars: bool = True) -> str:
    """Sanitize Unicode content by removing or replacing problematic characters.

    Args:
        content: Content to sanitize
        remove_control_chars: Whether to remove control characters

    Returns:
        str: Sanitized content
    """
    if remove_control_chars:
        # Remove control characters except newlines and tabs
        import re

        content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", content)

    # Replace common problematic characters
    replacements = {
        "\u2028": "\n",  # Line separator
        "\u2029": "\n",  # Paragraph separator
        "\uFEFF": "",  # Zero-width no-break space (BOM)
        "\u200B": "",  # Zero-width space
        "\u200C": "",  # Zero-width non-joiner
        "\u200D": "",  # Zero-width joiner
    }

    for old_char, new_char in replacements.items():
        content = content.replace(old_char, new_char)

    return content


def convert_to_utf8_with_fallback(file_path: str, original_encoding: str = None) -> bool:
    """Convert a file to UTF-8 encoding with robust fallback handling.

    Args:
        file_path: Path to the file to convert
        original_encoding: Original encoding of the file (if known)

    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Read with fallback encoding
        content, used_encoding = read_with_fallback_encoding(
            file_path, [original_encoding] if original_encoding else None
        )

        # Sanitize content
        sanitized_content = sanitize_unicode_content(content)

        # Write with UTF-8 encoding
        success = write_with_encoding_fallback(file_path, sanitized_content, "utf-8")

        if success:
            logger.info(f"Successfully converted {file_path} from {used_encoding} to UTF-8")
        else:
            logger.error(f"Failed to write {file_path} after conversion")

        return success
    except Exception as e:
        logger.error(f"Failed to convert {file_path}: {e}")
        return False


def scan_project_for_utf8(
    root_dir: str = ".", convert: bool = False, file_extensions: Optional[List[str]] = None
) -> List[Tuple[str, str]]:
    """Scan project files for non-UTF-8 encoding and optionally convert them.

    Args:
        root_dir: Root directory to start scanning from
        convert: Whether to automatically convert non-UTF-8 files
        file_extensions: List of file extensions to check (default: .py, .json, .csv, .txt, .md)

    Returns:
        List[Tuple[str, str]]: List of (file_path, encoding) tuples for non-UTF-8 files
    """
    if file_extensions is None:
        file_extensions = [".py", ".json", ".csv", ".txt", ".md"]

    non_utf8_files = []
    logger.info(f"Scanning {root_dir} for non-UTF-8 files...")

    try:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(dirpath, file)

                    # Skip binary files and certain directories
                    if any(skip_dir in file_path for skip_dir in [".git", "__pycache__", "venv"]):
                        continue

                    if not is_utf8(file_path):
                        encoding_info = detect_encoding(file_path)
                        encoding = encoding_info["encoding"]

                        if encoding:
                            non_utf8_files.append((file_path, encoding))
                            logger.warning(f"Non-UTF8 file found: {file_path} ({encoding})")

                            if convert:
                                success = convert_to_utf8_with_fallback(file_path, encoding)
                                if success:
                                    logger.info(f"Converted to UTF-8: {file_path}")
                                else:
                                    logger.error(f"Failed to convert: {file_path}")
                        else:
                            logger.error(f"Could not detect encoding for {file_path}")

    except Exception as e:
        logger.error(f"Error during scanning: {e}")

    if not non_utf8_files:
        logger.info("All scanned files are UTF-8 encoded.")
    else:
        logger.info(f"Found {len(non_utf8_files)} non-UTF-8 files.")
        if not convert:
            logger.info("Run with convert=True to automatically convert files to UTF-8.")

    return non_utf8_files


def main():
    """Main function to run the UTF-8 scanner."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan and convert files to UTF-8 encoding")
    parser.add_argument("--root-dir", default=".", help="Root directory to scan")
    parser.add_argument("--convert", action="store_true", help="Convert non-UTF-8 files")
    parser.add_argument("--extensions", nargs="+", help="File extensions to check")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--fallback", action="store_true", help="Use robust fallback encoding")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Run scanner
    non_utf8_files = scan_project_for_utf8(
        root_dir=args.root_dir, convert=args.convert, file_extensions=args.extensions
    )

    # Print summary
    if non_utf8_files:
        logger.info("\nNon-UTF-8 files found:")
        for file_path, encoding in non_utf8_files:
            logger.info(f"- {file_path} ({encoding})")


if __name__ == "__main__":
    main()
