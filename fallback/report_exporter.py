"""
Fallback Report Exporter Implementation

Provides fallback functionality for report generation and export when
the primary report exporter is unavailable.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FallbackReportExporter:
    """
    Fallback implementation of the Report Exporter.

    Provides basic report generation and export functionality when the
    primary report exporter is unavailable.
    """

    def __init__(self) -> None:
        """
        Initialize the fallback report exporter.

        Sets up basic logging and initializes export directories for
        fallback operations.
        """
        self._status = "fallback"
        self._export_dir = "reports/fallback"
        self._ensure_export_directory()
        logger.info("FallbackReportExporter initialized")

    def _ensure_export_directory(self) -> None:
        """
        Ensure the export directory exists.
        """
        try:
            os.makedirs(self._export_dir, exist_ok=True)
            logger.debug(f"Export directory ensured: {self._export_dir}")
        except Exception as e:
            logger.error(f"Error creating export directory: {e}")

    def export_report(self, data: Dict[str, Any], format: str = "json", filename: Optional[str] = None) -> str:
        """
        Export a report in the specified format (fallback implementation).

        Args:
            data: Data to export
            format: Export format (json, txt, csv)
            filename: Optional filename (auto-generated if not provided)

        Returns:
            str: Path to the exported file
        """
        try:
            logger.info(f"Exporting report in {format} format")

            # Generate filename if not provided
            if filename is None:
                # Use microsecond precision to avoid overwrites
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                filename = f"fallback_report_{timestamp}.{format}"

            filepath = os.path.join(self._export_dir, filename)

            # Add metadata
            export_data = {
                "data": data,
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "format": format,
                    "fallback_mode": True,
                    "exporter": "FallbackReportExporter",
                },
            }

            # Export based on format
            if format.lower() == "json":
                self._export_json(export_data, filepath)
            elif format.lower() == "txt":
                self._export_text(export_data, filepath)
            elif format.lower() == "csv":
                self._export_csv(export_data, filepath)
            else:
                logger.warning(f"Unsupported format: {format}, defaulting to JSON")
                self._export_json(export_data, filepath)

            logger.info(f"Report exported successfully: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return f"export_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def _export_json(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Export data as JSON.

        Args:
            data: Data to export
            filepath: Output file path
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            raise

    def _export_text(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Export data as text.

        Args:
            data: Data to export
            filepath: Output file path
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("FALLBACK REPORT EXPORT\n")
                f.write("=" * 50 + "\n\n")

                # Write metadata
                metadata = data.get("metadata", {})
                f.write("METADATA:\n")
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

                # Write main data
                main_data = data.get("data", {})
                f.write("REPORT DATA:\n")
                f.write("-" * 30 + "\n")

                self._write_dict_to_text(main_data, f, indent=0)

        except Exception as e:
            logger.error(f"Error exporting text: {e}")
            raise

    def _write_dict_to_text(self, data: Any, file, indent: int = 0) -> None:
        """
        Write dictionary data to text file with proper formatting.

        Args:
            data: Data to write
            file: File object to write to
            indent: Indentation level
        """
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    file.write("  " * indent + f"{key}: ")
                    if isinstance(value, (dict, list)):
                        file.write("\n")
                        self._write_dict_to_text(value, file, indent + 1)
                    else:
                        file.write(f"{value}\n")
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    file.write("  " * indent + f"[{i}]: ")
                    if isinstance(item, (dict, list)):
                        file.write("\n")
                        self._write_dict_to_text(item, file, indent + 1)
                    else:
                        file.write(f"{item}\n")
            else:
                file.write(f"{data}\n")

        except Exception as e:
            logger.error(f"Error writing data to text: {e}")

    def _export_csv(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Export data as CSV (simplified implementation).

        Args:
            data: Data to export
            filepath: Output file path
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("key,value\n")

                # Flatten data for CSV export
                flattened = self._flatten_dict(data)
                for key, value in flattened.items():
                    f.write(f'"{key}","{value}"\n')

        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            raise

    def _flatten_dict(self, data: Any, prefix: str = "") -> Dict[str, str]:
        """
        Flatten nested dictionary for CSV export.

        Args:
            data: Data to flatten
            prefix: Key prefix for nested items

        Returns:
            Dict[str, str]: Flattened dictionary
        """
        try:
            flattened = {}

            if isinstance(data, dict):
                for key, value in data.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (dict, list)):
                        flattened.update(self._flatten_dict(value, new_key))
                    else:
                        flattened[new_key] = str(value)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    new_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
                    if isinstance(item, (dict, list)):
                        flattened.update(self._flatten_dict(item, new_key))
                    else:
                        flattened[new_key] = str(item)
            else:
                flattened[prefix] = str(data)

            return flattened

        except Exception as e:
            logger.error(f"Error flattening dictionary: {e}")
            return {}

    def generate_summary_report(self, data: Dict[str, Any]) -> str:
        """
        Generate a summary report (fallback implementation).

        Args:
            data: Data to summarize

        Returns:
            str: Summary report content
        """
        try:
            logger.info("Generating summary report using fallback exporter")

            summary = []
            summary.append("FALLBACK SUMMARY REPORT")
            summary.append("=" * 50)
            summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary.append("")

            # Extract key metrics
            if "forecast" in data:
                forecast_data = data["forecast"]
                summary.append("FORECAST SUMMARY:")
                summary.append(f"  Symbol: {forecast_data.get('symbol', 'Unknown')}")
                summary.append(f"  Model: {forecast_data.get('model', 'Unknown')}")
                summary.append(f"  Confidence: {forecast_data.get('confidence', 0):.1%}")
                summary.append("")

            if "strategy" in data:
                strategy_data = data["strategy"]
                summary.append("STRATEGY SUMMARY:")
                summary.append(f"  Strategy: {strategy_data.get('strategy', 'Unknown')}")
                summary.append(f"  Signal: {strategy_data.get('signal', 'Unknown')}")
                summary.append(f"  Performance: {strategy_data.get('performance', 'Unknown')}")
                summary.append("")

            if "portfolio" in data:
                portfolio_data = data["portfolio"]
                summary.append("PORTFOLIO SUMMARY:")
                summary.append(f"  Total Value: ${portfolio_data.get('total_value', 0):,.2f}")
                summary.append(f"  Positions: {len(portfolio_data.get('positions', []))}")
                summary.append(f"  Cash: ${portfolio_data.get('cash', 0):,.2f}")
                summary.append("")

            summary.append("NOTE: This is a fallback report generated due to system limitations.")

            return "\n".join(summary)

        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return f"Error generating summary report: {str(e)}"

    def get_export_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get export history (fallback implementation).

        Args:
            limit: Maximum number of exports to return

        Returns:
            List[Dict[str, Any]]: Export history
        """
        try:
            logger.info(f"Getting export history (limit: {limit})")

            history = []

            if os.path.exists(self._export_dir):
                files = os.listdir(self._export_dir)
                files.sort(key=lambda x: os.path.getmtime(os.path.join(self._export_dir, x)), reverse=True)

                for filename in files[:limit]:
                    filepath = os.path.join(self._export_dir, filename)
                    stat = os.stat(filepath)

                    history.append(
                        {
                            "filename": filename,
                            "filepath": filepath,
                            "size": stat.st_size,
                            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        }
                    )

            return history

        except Exception as e:
            logger.error(f"Error getting export history: {e}")
            return []

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the health status of the fallback report exporter.

        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            export_count = len(os.listdir(self._export_dir)) if os.path.exists(self._export_dir) else 0

            return {
                "status": self._status,
                "export_directory": self._export_dir,
                "total_exports": export_count,
                "directory_exists": os.path.exists(self._export_dir),
                "fallback_mode": True,
                "message": "Using fallback report exporter",
            }
        except Exception as e:
            logger.error(f"Error getting fallback report exporter health: {e}")
            return {
                "status": "error",
                "export_directory": self._export_dir,
                "total_exports": 0,
                "fallback_mode": True,
                "error": str(e),
            }
