"""
Report Exporter for Trade Reports

Provides functionality to export trade reports in various formats including CSV and PDF.
"""

import csv
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logging.warning("FPDF not available. PDF export will be disabled.")

logger = logging.getLogger(__name__)


class ReportExporter:
    """
    Export trade reports in various formats.
    """

    def __init__(self, export_dir: str = "reports/exports"):
        """
        Initialize the report exporter.

        Args:
            export_dir: Directory to save exported files
        """
        self.export_dir = export_dir
        self._ensure_export_directory()

    def _ensure_export_directory(self) -> None:
        """Ensure the export directory exists."""
        try:
            os.makedirs(self.export_dir, exist_ok=True)
            logger.debug(f"Export directory ensured: {self.export_dir}")
        except Exception as e:
            logger.error(f"Error creating export directory: {e}")

    def export_trade_report(
        self,
        signals: Union[List[Dict], pd.DataFrame],
        format: str = "CSV",
        filename: Optional[str] = None,
        include_summary: bool = True
    ) -> str:
        """
        Export trade report in the specified format.

        Args:
            signals: Trade signals data (list of dicts or DataFrame)
            format: Export format ('CSV' or 'PDF')
            filename: Optional filename (auto-generated if not provided)
            include_summary: Whether to include summary statistics

        Returns:
            str: Path to the exported file
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(signals, list):
                df = pd.DataFrame(signals)
            else:
                df = signals.copy()

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trade_report_{timestamp}.{format.lower()}"

            filepath = os.path.join(self.export_dir, filename)

            # Export based on format
            if format.upper() == "CSV":
                return self.export_to_csv(df, filepath, include_summary)
            elif format.upper() == "PDF":
                if not FPDF_AVAILABLE:
                    raise ImportError("FPDF not available. Install with: pip install fpdf")
                return self.export_to_pdf(df, filepath, include_summary)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'CSV' or 'PDF'")

        except Exception as e:
            logger.error(f"Error exporting trade report: {e}")
            raise

    def export_to_csv(
        self,
        df: pd.DataFrame,
        filepath: str,
        include_summary: bool = True
    ) -> str:
        """
        Export trade signals to CSV format.

        Args:
            df: DataFrame containing trade signals
            filepath: Output file path
            include_summary: Whether to include summary statistics

        Returns:
            str: Path to the exported file
        """
        try:
            # Export main signals data
            df.to_csv(filepath, index=False)

            # Add summary statistics if requested
            if include_summary:
                summary_filepath = filepath.replace('.csv', '_summary.csv')
                summary_stats = self._calculate_summary_stats(df)
                summary_df = pd.DataFrame([summary_stats])
                summary_df.to_csv(summary_filepath, index=False)
                logger.info(f"Summary exported to: {summary_filepath}")

            logger.info(f"Trade report exported to CSV: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise

    def export_to_pdf(
        self,
        df: pd.DataFrame,
        filepath: str,
        include_summary: bool = True
    ) -> str:
        """
        Export trade signals to PDF format using FPDF.

        Args:
            df: DataFrame containing trade signals
            filepath: Output file path
            include_summary: Whether to include summary statistics

        Returns:
            str: Path to the exported file
        """
        try:
            pdf = FPDF()
            pdf.add_page()

            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Trade Report', ln=True, align='C')
            pdf.ln(5)

            # Summary statistics
            if include_summary:
                summary_stats = self._calculate_summary_stats(df)
                self._add_summary_section(pdf, summary_stats)
                pdf.ln(10)

            # Trade signals table
            self._add_signals_table(pdf, df)

            # Save PDF
            pdf.output(filepath)
            logger.info(f"Trade report exported to PDF: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            raise

    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics from trade signals.

        Args:
            df: DataFrame containing trade signals

        Returns:
            Dict containing summary statistics
        """
        try:
            stats = {}

            # Basic counts
            stats['total_trades'] = len(df)
            stats['winning_trades'] = len(df[df.get('pnl', 0) > 0]) if 'pnl' in df.columns else 0
            stats['losing_trades'] = len(df[df.get('pnl', 0) < 0]) if 'pnl' in df.columns else 0

            # Win rate
            if stats['total_trades'] > 0:
                stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
            else:
                stats['win_rate'] = 0.0

            # PnL statistics
            if 'pnl' in df.columns:
                pnl_series = df['pnl'].dropna()
                if len(pnl_series) > 0:
                    stats['total_pnl'] = pnl_series.sum()
                    stats['avg_pnl'] = pnl_series.mean()
                    stats['max_profit'] = pnl_series.max()
                    stats['max_loss'] = pnl_series.min()
                    stats['pnl_std'] = pnl_series.std()

                    # Sharpe ratio (simplified)
                    if stats['pnl_std'] > 0:
                        stats['sharpe_ratio'] = stats['avg_pnl'] / stats['pnl_std']
                    else:
                        stats['sharpe_ratio'] = 0.0

                    # Max drawdown calculation
                    cumulative = pnl_series.cumsum()
                    running_max = cumulative.expanding().max()
                    drawdown = cumulative - running_max
                    stats['max_drawdown'] = drawdown.min()
                else:
                    stats.update({
                        'total_pnl': 0.0,
                        'avg_pnl': 0.0,
                        'max_profit': 0.0,
                        'max_loss': 0.0,
                        'pnl_std': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0
                    })
            else:
                stats.update({
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0,
                    'pnl_std': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0
                })

            # Date range
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    stats['start_date'] = df['timestamp'].min().strftime('%Y-%m-%d')
                    stats['end_date'] = df['timestamp'].max().strftime('%Y-%m-%d')
                except:
                    stats['start_date'] = 'N/A'
                    stats['end_date'] = 'N/A'
            else:
                stats['start_date'] = 'N/A'
                stats['end_date'] = 'N/A'

            return stats

        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return {}

    def _add_summary_section(self, pdf: FPDF, stats: Dict[str, Any]) -> None:
        """
        Add summary statistics section to PDF.

        Args:
            pdf: FPDF object
            stats: Summary statistics dictionary
        """
        try:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Summary Statistics', ln=True)
            pdf.ln(2)

            pdf.set_font('Arial', '', 10)

            # Performance metrics
            metrics = [
                ('Total Trades', f"{stats.get('total_trades', 0)}"),
                ('Win Rate', f"{stats.get('win_rate', 0):.2%}"),
                ('Total PnL', f"${stats.get('total_pnl', 0):,.2f}"),
                ('Average PnL', f"${stats.get('avg_pnl', 0):,.2f}"),
                ('Sharpe Ratio', f"{stats.get('sharpe_ratio', 0):.2f}"),
                ('Max Drawdown', f"${stats.get('max_drawdown', 0):,.2f}"),
                ('Max Profit', f"${stats.get('max_profit', 0):,.2f}"),
                ('Max Loss', f"${stats.get('max_loss', 0):,.2f}"),
            ]

            for i, (label, value) in enumerate(metrics):
                pdf.cell(60, 8, f"{label}: {value}")
                if i % 2 == 1:  # New line every 2 items
                    pdf.ln()

            if len(metrics) % 2 == 1:  # Add new line if odd number of items
                pdf.ln()

        except Exception as e:
            logger.error(f"Error adding summary section: {e}")

    def _add_signals_table(self, pdf: FPDF, df: pd.DataFrame) -> None:
        """
        Add trade signals table to PDF.

        Args:
            pdf: FPDF object
            df: DataFrame containing trade signals
        """
        try:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Trade Signals', ln=True)
            pdf.ln(2)

            # Determine columns to include
            important_cols = ['timestamp', 'symbol', 'signal', 'price', 'pnl', 'strategy']
            available_cols = [col for col in important_cols if col in df.columns]

            # Add any other columns that might be useful
            other_cols = [col for col in df.columns if col not in important_cols and col not in ['index']]
            display_cols = available_cols + other_cols[:3]  # Limit to 3 additional columns

            # Set up table
            col_widths = [40, 25, 20, 25, 25, 30]  # Default widths
            if len(display_cols) < len(col_widths):
                col_widths = col_widths[:len(display_cols)]
            elif len(display_cols) > len(col_widths):
                col_widths.extend([25] * (len(display_cols) - len(col_widths)))

            # Header
            pdf.set_font('Arial', 'B', 8)
            for i, col in enumerate(display_cols):
                pdf.cell(col_widths[i], 8, str(col).title(), border=1)
            pdf.ln()

            # Data rows (limit to first 20 rows to avoid PDF overflow)
            pdf.set_font('Arial', '', 7)
            for idx, row in df.head(20).iterrows():
                for i, col in enumerate(display_cols):
                    value = str(row.get(col, ''))[:20]  # Truncate long values
                    pdf.cell(col_widths[i], 6, value, border=1)
                pdf.ln()

            # Add note if truncated
            if len(df) > 20:
                pdf.ln(5)
                pdf.set_font('Arial', 'I', 8)
                pdf.cell(0, 6, f"Note: Showing first 20 of {len(df)} trades", ln=True)

        except Exception as e:
            logger.error(f"Error adding signals table: {e}")


def export_trade_report(
    signals: Union[List[Dict], pd.DataFrame],
    format: str = "CSV",
    export_dir: str = "reports/exports",
    filename: Optional[str] = None,
    include_summary: bool = True
) -> str:
    """
    Convenience function to export trade report.

    Args:
        signals: Trade signals data
        format: Export format ('CSV' or 'PDF')
        export_dir: Directory to save exported files
        filename: Optional filename
        include_summary: Whether to include summary statistics

    Returns:
        str: Path to the exported file
    """
    exporter = ReportExporter(export_dir)
    return exporter.export_trade_report(signals, format, filename, include_summary)
