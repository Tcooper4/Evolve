"""
Export Report Module

Enhanced with Batch 10 features: comprehensive error handling for all file writing operations
with meaningful error messages and support for zipped exports and visual HTML reports.
"""

import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ExportReport:
    """Enhanced report export with comprehensive error handling."""

    def __init__(self, output_dir: str = "reports"):
        """Initialize the export report handler.
        
        Args:
            output_dir: Directory for exported reports
        """
        self.output_dir = Path(output_dir)
        self.export_history = []
        
        # Create output directory with error handling
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Export directory initialized: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create export directory {self.output_dir}: {e}")
            raise RuntimeError(f"Cannot initialize export system: {e}")

    def export_to_csv(self, data: Union[pd.DataFrame, List[Dict]], filename: str) -> Dict[str, Any]:
        """Export data to CSV with comprehensive error handling.
        
        Args:
            data: Data to export (DataFrame or list of dictionaries)
            filename: Output filename
            
        Returns:
            Dictionary with export result and error information
        """
        try:
            filepath = self.output_dir / filename
            
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Validate data
            if df.empty:
                return {
                    "success": False,
                    "error": "No data to export",
                    "filepath": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            
            # Verify file was created
            if not filepath.exists():
                return {
                    "success": False,
                    "error": "File was not created after export",
                    "filepath": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log successful export
            export_record = {
                "timestamp": datetime.now().isoformat(),
                "format": "csv",
                "filename": filename,
                "filepath": str(filepath),
                "rows": len(df),
                "columns": len(df.columns),
                "file_size": filepath.stat().st_size
            }
            self.export_history.append(export_record)
            
            logger.info(f"CSV export successful: {filepath} ({len(df)} rows)")
            
            return {
                "success": True,
                "filepath": str(filepath),
                "rows": len(df),
                "columns": len(df.columns),
                "file_size": filepath.stat().st_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except PermissionError as e:
            error_msg = f"Permission denied writing to {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except OSError as e:
            error_msg = f"OS error writing CSV file {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Unexpected error exporting CSV {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }

    def export_to_json(self, data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export data to JSON with comprehensive error handling.
        
        Args:
            data: Data to export
            filename: Output filename
            
        Returns:
            Dictionary with export result and error information
        """
        try:
            filepath = self.output_dir / filename
            
            # Validate data
            if not data:
                return {
                    "success": False,
                    "error": "No data to export",
                    "filepath": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Test serialization first
            try:
                json.dumps(data, default=str)
            except (TypeError, ValueError) as e:
                return {
                    "success": False,
                    "error": f"Data serialization error: {e}",
                    "filepath": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Export to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            # Verify file was created
            if not filepath.exists():
                return {
                    "success": False,
                    "error": "File was not created after export",
                    "filepath": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log successful export
            export_record = {
                "timestamp": datetime.now().isoformat(),
                "format": "json",
                "filename": filename,
                "filepath": str(filepath),
                "file_size": filepath.stat().st_size
            }
            self.export_history.append(export_record)
            
            logger.info(f"JSON export successful: {filepath}")
            
            return {
                "success": True,
                "filepath": str(filepath),
                "file_size": filepath.stat().st_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except PermissionError as e:
            error_msg = f"Permission denied writing to {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except OSError as e:
            error_msg = f"OS error writing JSON file {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except TypeError as e:
            error_msg = f"Data serialization error for JSON {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Unexpected error exporting JSON {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }

    def export_to_html(self, data: Dict[str, Any], filename: str, include_charts: bool = True) -> Dict[str, Any]:
        """Export data to HTML with comprehensive error handling.
        
        Args:
            data: Data to export
            filename: Output filename
            include_charts: Whether to include interactive charts
            
        Returns:
            Dictionary with export result and error information
        """
        try:
            filepath = self.output_dir / filename
            
            # Generate HTML content
            html_content = self._generate_html_report(data, include_charts)
            
            # Write HTML file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Verify file was created
            if not filepath.exists():
                return {
                    "success": False,
                    "error": "File was not created after export",
                    "filepath": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log successful export
            export_record = {
                "timestamp": datetime.now().isoformat(),
                "format": "html",
                "filename": filename,
                "filepath": str(filepath),
                "file_size": filepath.stat().st_size,
                "includes_charts": include_charts
            }
            self.export_history.append(export_record)
            
            logger.info(f"HTML export successful: {filepath}")
            
            return {
                "success": True,
                "filepath": str(filepath),
                "file_size": filepath.stat().st_size,
                "includes_charts": include_charts,
                "timestamp": datetime.now().isoformat()
            }
            
        except PermissionError as e:
            error_msg = f"Permission denied writing to {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except OSError as e:
            error_msg = f"OS error writing HTML file {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Unexpected error exporting HTML {filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }

    def create_zipped_export(self, files: List[str], zip_filename: str) -> Dict[str, Any]:
        """Create zipped export bundle with comprehensive error handling.
        
        Args:
            files: List of file paths to include in zip
            zip_filename: Output zip filename
            
        Returns:
            Dictionary with export result and error information
        """
        try:
            zip_path = self.output_dir / zip_filename
            
            # Validate input files
            valid_files = []
            for file_path in files:
                if Path(file_path).exists():
                    valid_files.append(file_path)
                else:
                    logger.warning(f"File not found for zip export: {file_path}")
            
            if not valid_files:
                return {
                    "success": False,
                    "error": "No valid files found for zip export",
                    "filepath": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Create zip file
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in valid_files:
                    try:
                        zipf.write(file_path, Path(file_path).name)
                    except Exception as e:
                        logger.warning(f"Failed to add {file_path} to zip: {e}")
            
            # Verify zip was created
            if not zip_path.exists():
                return {
                    "success": False,
                    "error": "Zip file was not created",
                    "filepath": None,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log successful export
            export_record = {
                "timestamp": datetime.now().isoformat(),
                "format": "zip",
                "filename": zip_filename,
                "filepath": str(zip_path),
                "file_size": zip_path.stat().st_size,
                "files_included": len(valid_files)
            }
            self.export_history.append(export_record)
            
            logger.info(f"Zipped export successful: {zip_path} ({len(valid_files)} files)")
            
            return {
                "success": True,
                "filepath": str(zip_path),
                "file_size": zip_path.stat().st_size,
                "files_included": len(valid_files),
                "timestamp": datetime.now().isoformat()
            }
            
        except PermissionError as e:
            error_msg = f"Permission denied creating zip file {zip_filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except OSError as e:
            error_msg = f"OS error creating zip file {zip_filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Unexpected error creating zip export {zip_filename}: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "filepath": None,
                "timestamp": datetime.now().isoformat()
            }

    def _generate_html_report(self, data: Dict[str, Any], include_charts: bool = True) -> str:
        """Generate HTML report content."""
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; min-width: 150px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .chart-container {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }}
        .error {{ color: #dc3545; background: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .success {{ color: #155724; background: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #007bff; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
    {chart_scripts}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trading Report</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        {content}
    </div>
</body>
</html>"""
        
        # Generate content sections
        content_sections = []
        
        # Summary section
        if "summary" in data:
            summary_html = self._generate_summary_section(data["summary"])
            content_sections.append(summary_html)
        
        # Performance metrics section
        if "performance" in data:
            performance_html = self._generate_performance_section(data["performance"])
            content_sections.append(performance_html)
        
        # Trade data section
        if "trades" in data:
            trades_html = self._generate_trades_section(data["trades"])
            content_sections.append(trades_html)
        
        # Charts section
        if include_charts and "charts" in data:
            charts_html = self._generate_charts_section(data["charts"])
            content_sections.append(charts_html)
        
        # Combine all sections
        content = "\n".join(content_sections)
        
        # Add chart scripts if needed
        chart_scripts = ""
        if include_charts:
            chart_scripts = """
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                // Chart initialization code would go here
                console.log('Charts loaded');
            </script>
            """
        
        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content=content,
            chart_scripts=chart_scripts
        )

    def _generate_summary_section(self, summary: Dict[str, Any]) -> str:
        """Generate summary section HTML."""
        html = """
        <div class="section">
            <h2>Summary</h2>
            <div class="metrics">
        """
        
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
            else:
                formatted_value = str(value)
            
            html += f"""
                <div class="metric">
                    <div class="metric-value">{formatted_value}</div>
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        return html

    def _generate_performance_section(self, performance: Dict[str, Any]) -> str:
        """Generate performance section HTML."""
        html = """
        <div class="section">
            <h2>Performance Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for key, value in performance.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{formatted_value}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        return html

    def _generate_trades_section(self, trades: List[Dict[str, Any]]) -> str:
        """Generate trades section HTML."""
        if not trades:
            return """
            <div class="section">
                <h2>Trades</h2>
                <p>No trade data available.</p>
            </div>
            """
        
        html = """
        <div class="section">
            <h2>Trades</h2>
            <table>
                <thead>
                    <tr>
        """
        
        # Generate headers from first trade
        headers = list(trades[0].keys())
        for header in headers:
            html += f"<th>{header.replace('_', ' ').title()}</th>"
        
        html += """
                </tr>
                </thead>
                <tbody>
        """
        
        # Generate rows
        for trade in trades:
            html += "<tr>"
            for header in headers:
                value = trade.get(header, "")
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                html += f"<td>{formatted_value}</td>"
            html += "</tr>"
        
        html += """
                </tbody>
            </table>
        </div>
        """
        return html

    def _generate_charts_section(self, charts: Dict[str, Any]) -> str:
        """Generate charts section HTML."""
        html = """
        <div class="section">
            <h2>Charts</h2>
        """
        
        for chart_name, chart_data in charts.items():
            html += f"""
            <div class="chart-container">
                <h3>{chart_name.replace('_', ' ').title()}</h3>
                <div id="chart_{chart_name}"></div>
            </div>
            """
        
        html += "</div>"
        return html

    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of all exports."""
        if not self.export_history:
            return {"total_exports": 0}
        
        format_counts = {}
        total_size = 0
        
        for export in self.export_history:
            format_type = export["format"]
            format_counts[format_type] = format_counts.get(format_type, 0) + 1
            total_size += export.get("file_size", 0)
        
        return {
            "total_exports": len(self.export_history),
            "format_counts": format_counts,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "recent_exports": self.export_history[-10:] if len(self.export_history) > 10 else self.export_history
        }

    def cleanup_old_exports(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Clean up old export files.
        
        Args:
            days_to_keep: Number of days to keep files
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            deleted_files = []
            deleted_size = 0
            
            for file_path in self.output_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_date:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_files.append(str(file_path))
                        deleted_size += file_size
                        logger.info(f"Deleted old export file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
            
            return {
                "success": True,
                "deleted_files": deleted_files,
                "deleted_count": len(deleted_files),
                "deleted_size_bytes": deleted_size,
                "deleted_size_mb": deleted_size / (1024 * 1024)
            }
            
        except Exception as e:
            error_msg = f"Error during cleanup: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
