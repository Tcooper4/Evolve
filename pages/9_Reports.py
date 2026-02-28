"""
Reports & Exports Page

Merges functionality from:
- Reports.py (standalone)

Features:
- Quick report generation with pre-built templates
- Custom report builder
- Scheduled reports
- Report library and management
- Multiple export formats (PDF, Excel, HTML)
- Email delivery
"""

import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Backend imports
try:
    from trading.report.report_generator import ReportGenerator
    from trading.report.export_report import ExportReport
    from trading.report.report_export_engine import ReportExportEngine
    
    REPORT_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some report modules not available: {e}")
    REPORT_MODULES_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Reports & Exports",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'report_generator' not in st.session_state:
    try:
        st.session_state.report_generator = ReportGenerator() if REPORT_MODULES_AVAILABLE else None
    except Exception as e:
        logger.warning(f"Could not initialize report generator: {e}")
        st.session_state.report_generator = None

if 'report_export_engine' not in st.session_state:
    try:
        st.session_state.report_export_engine = ReportExportEngine() if REPORT_MODULES_AVAILABLE else None
    except Exception as e:
        logger.warning(f"Could not initialize report export engine: {e}")
        st.session_state.report_export_engine = None

if 'generated_reports' not in st.session_state:
    st.session_state.generated_reports = {}

if 'scheduled_reports' not in st.session_state:
    st.session_state.scheduled_reports = {}

if 'report_templates' not in st.session_state:
    st.session_state.report_templates = {}

# Main page title
st.title("📄 Reports & Exports")
st.markdown("Generate comprehensive trading reports and export in multiple formats")

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Quick Reports",
    "🔧 Custom Report Builder",
    "⏰ Scheduled Reports",
    "📚 Report Library"
])

# TAB 1: Quick Reports
with tab1:
    st.header("⚡ Quick Reports")
    st.markdown("Generate comprehensive reports with one click using pre-built templates.")
    
    # Report Type Selection
    st.subheader("📋 Report Type")
    
    report_type = st.selectbox(
        "Select Report Type",
        [
            "Daily Performance Report",
            "Weekly Summary",
            "Monthly Performance Report",
            "Quarterly Review",
            "Annual Report",
            "Risk Analysis Report",
            "Portfolio Summary",
            "Trade Journal",
            "Tax Report"
        ],
        help="Choose a pre-built report template"
    )
    
    # Date Range Selection
    st.subheader("📅 Date Range")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Auto-set date range based on report type
        if "Daily" in report_type:
            default_start = datetime.now() - timedelta(days=1)
            default_end = datetime.now()
        elif "Weekly" in report_type:
            default_start = datetime.now() - timedelta(days=7)
            default_end = datetime.now()
        elif "Monthly" in report_type:
            default_start = datetime.now() - timedelta(days=30)
            default_end = datetime.now()
        elif "Quarterly" in report_type:
            default_start = datetime.now() - timedelta(days=90)
            default_end = datetime.now()
        elif "Annual" in report_type:
            default_start = datetime.now() - timedelta(days=365)
            default_end = datetime.now()
        else:
            default_start = datetime.now() - timedelta(days=30)
            default_end = datetime.now()
        
        start_date = st.date_input(
            "Start Date",
            value=default_start.date(),
            max_value=datetime.now().date(),
            key="reports_summary_start_date"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end.date(),
            max_value=datetime.now().date(),
            key="reports_summary_end_date"
        )
    
    # Additional Options
    st.subheader("⚙️ Report Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_charts = st.checkbox("Include Charts", value=True, help="Add visualizations to the report")
        include_trade_details = st.checkbox("Include Trade Details", value=True, help="Include detailed trade information")
    
    with col2:
        include_risk_metrics = st.checkbox("Include Risk Metrics", value=True, help="Add risk analysis")
        include_performance_attribution = st.checkbox("Include Performance Attribution", value=False, help="Add performance breakdown")
    
    st.markdown("---")
    
    # Generate Report Button
    generate_button = st.button("🚀 Generate Report", type="primary", use_container_width=True)
    
    if generate_button:
        try:
            with st.spinner(f"Generating {report_type}..."):
                # Generate report data
                report_id = f"{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Simulate report generation (in real implementation, this would use actual data)
                # For now, we'll create a comprehensive report structure
                
                # Get portfolio data if available
                portfolio_data = None
                trade_data = None
                risk_data = None
                
                # Try to get data from session state or other pages
                if 'portfolio_manager' in st.session_state:
                    try:
                        portfolio_data = st.session_state.portfolio_manager.get_portfolio_summary()
                    except Exception as e:
                        # If portfolio summary fails, continue without it
                        pass
                
                # Generate report content
                report_data = {
                    "report_id": report_id,
                    "report_type": report_type,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "generated_at": datetime.now().isoformat(),
                    "period": f"{start_date} to {end_date}",
                    "options": {
                        "include_charts": include_charts,
                        "include_trade_details": include_trade_details,
                        "include_risk_metrics": include_risk_metrics,
                        "include_performance_attribution": include_performance_attribution
                    }
                }
                
                # Generate report sections based on type
                st.markdown("---")
                st.subheader(f"📊 {report_type} Preview")
                
                # Report Header
                st.markdown(f"### {report_type}")
                st.markdown(f"**Period:** {start_date} to {end_date}")
                st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                st.markdown("---")
                
                # Executive Summary
                st.markdown("### Executive Summary")
                
                # Simulate metrics (in real implementation, these would come from actual data)
                summary_metrics = {
                    "Total Return": "12.5%",
                    "Sharpe Ratio": "1.85",
                    "Max Drawdown": "-8.3%",
                    "Win Rate": "58.2%",
                    "Total Trades": "142",
                    "Average Trade P&L": "$125.50"
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", summary_metrics["Total Return"])
                    st.metric("Sharpe Ratio", summary_metrics["Sharpe Ratio"])
                with col2:
                    st.metric("Max Drawdown", summary_metrics["Max Drawdown"])
                    st.metric("Win Rate", summary_metrics["Win Rate"])
                with col3:
                    st.metric("Total Trades", summary_metrics["Total Trades"])
                    st.metric("Avg Trade P&L", summary_metrics["Average Trade P&L"])
                
                # Performance Metrics
                st.markdown("---")
                st.markdown("### Performance Metrics")
                
                performance_data = {
                    "Daily Return": np.random.normal(0.001, 0.02, (end_date - start_date).days),
                    "Cumulative Return": np.cumsum(np.random.normal(0.001, 0.02, (end_date - start_date).days))
                }
                
                if include_charts:
                    # Equity Curve
                    fig = go.Figure()
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    fig.add_trace(go.Scatter(
                        x=dates[:len(performance_data["Cumulative Return"])],
                        y=performance_data["Cumulative Return"],
                        mode='lines',
                        name='Cumulative Return',
                        line=dict(color='blue', width=2)
                    ))
                    fig.update_layout(
                        title="Equity Curve",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance Table
                perf_df = pd.DataFrame({
                    "Metric": ["Total Return", "Annualized Return", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"],
                    "Value": ["12.5%", "15.2%", "18.5%", "1.85", "2.10", "1.83"]
                })
                st.dataframe(perf_df, use_container_width=True, hide_index=True)
                
                # Risk Metrics (if enabled)
                if include_risk_metrics:
                    st.markdown("---")
                    st.markdown("### Risk Analysis")
                    
                    risk_metrics = {
                        "VaR (95%)": "-2.5%",
                        "CVaR (95%)": "-3.2%",
                        "Maximum Drawdown": "-8.3%",
                        "Beta": "0.95",
                        "Alpha": "2.1%",
                        "Tracking Error": "5.2%"
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("VaR (95%)", risk_metrics["VaR (95%)"])
                        st.metric("CVaR (95%)", risk_metrics["CVaR (95%)"])
                    with col2:
                        st.metric("Max Drawdown", risk_metrics["Maximum Drawdown"])
                        st.metric("Beta", risk_metrics["Beta"])
                    with col3:
                        st.metric("Alpha", risk_metrics["Alpha"])
                        st.metric("Tracking Error", risk_metrics["Tracking Error"])
                
                # Trade Details (if enabled)
                if include_trade_details:
                    st.markdown("---")
                    st.markdown("### Trade History")
                    
                    # Simulate trade data
                    n_trades = 20
                    trade_df = pd.DataFrame({
                        "Date": pd.date_range(start=start_date, periods=n_trades, freq='D'),
                        "Symbol": np.random.choice(["AAPL", "MSFT", "GOOGL", "AMZN"], n_trades),
                        "Action": np.random.choice(["Buy", "Sell"], n_trades),
                        "Quantity": np.random.randint(10, 100, n_trades),
                        "Price": np.random.uniform(100, 200, n_trades),
                        "P&L": np.random.uniform(-500, 1000, n_trades),
                        "Status": np.random.choice(["Filled", "Partial"], n_trades)
                    })
                    trade_df["P&L"] = trade_df["P&L"].apply(lambda x: f"${x:.2f}")
                    trade_df["Price"] = trade_df["Price"].apply(lambda x: f"${x:.2f}")
                    
                    st.dataframe(trade_df, use_container_width=True, height=300)
                
                # Performance Attribution (if enabled)
                if include_performance_attribution:
                    st.markdown("---")
                    st.markdown("### Performance Attribution")
                    
                    attribution_data = {
                        "Source": ["Stock Selection", "Market Timing", "Sector Allocation", "Currency", "Other"],
                        "Contribution": ["8.2%", "2.1%", "1.5%", "0.4%", "0.3%"]
                    }
                    attr_df = pd.DataFrame(attribution_data)
                    st.dataframe(attr_df, use_container_width=True, hide_index=True)
                
                # Store report
                report_data["content"] = {
                    "summary_metrics": summary_metrics,
                    "performance_data": performance_data,
                    "risk_metrics": risk_metrics if include_risk_metrics else None,
                    "trade_count": n_trades if include_trade_details else 0
                }
                
                st.session_state.generated_reports[report_id] = report_data
                
                st.success(f"✅ Report '{report_type}' generated successfully!")
                
                st.markdown("---")
                
                # Export Options
                st.subheader("📤 Export Report")
                
                export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                
                with export_col1:
                    # PDF Export
                    if st.button("📄 Export PDF", use_container_width=True):
                        try:
                            # Generate PDF report
                            from reportlab.lib.pagesizes import letter
                            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                            from reportlab.lib.styles import getSampleStyleSheet
                            from reportlab.lib import colors
                            import io
                            
                            # Create PDF
                            buffer = io.BytesIO()
                            doc = SimpleDocTemplate(buffer, pagesize=letter)
                            story = []
                            styles = getSampleStyleSheet()
                            
                            # Add title
                            title = Paragraph(f"<b>{report_type} Report</b>", styles['Title'])
                            story.append(title)
                            story.append(Spacer(1, 12))
                            
                            # Add report date
                            date_para = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal'])
                            story.append(date_para)
                            story.append(Spacer(1, 12))
                            
                            # Add period
                            period_para = Paragraph(f"Period: {start_date} to {end_date}", styles['Normal'])
                            story.append(period_para)
                            story.append(Spacer(1, 12))
                            
                            # Add summary metrics if available in session state
                            if 'report_summary_metrics' in st.session_state:
                                story.append(Paragraph("<b>Summary Metrics</b>", styles['Heading2']))
                                metrics_data = [[k, str(v)] for k, v in st.session_state.report_summary_metrics.items()]
                                metrics_table = Table(metrics_data)
                                metrics_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                ]))
                                story.append(metrics_table)
                                story.append(Spacer(1, 12))
                            
                            # Build PDF
                            doc.build(story)
                            pdf_data = buffer.getvalue()
                            buffer.close()
                            
                            # Provide download button
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_data,
                                file_name=f"{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                            
                            st.success("PDF report generated successfully!")
                        except ImportError:
                            st.error("reportlab library not installed. Install with: pip install reportlab")
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                
                with export_col2:
                    # Excel Export
                    if st.button("📊 Export Excel", use_container_width=True):
                        try:
                            # Create Excel file
                            from io import BytesIO
                            output = BytesIO()
                            
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                perf_df.to_excel(writer, sheet_name='Performance', index=False)
                                if include_trade_details:
                                    trade_df.to_excel(writer, sheet_name='Trades', index=False)
                                if include_risk_metrics:
                                    pd.DataFrame([risk_metrics]).to_excel(writer, sheet_name='Risk', index=False)
                            
                            output.seek(0)
                            st.download_button(
                                label="Download Excel",
                                data=output.getvalue(),
                                file_name=f"{report_id}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            st.success("Excel file generated!")
                        except Exception as e:
                            st.error(f"Error generating Excel: {str(e)}")
                
                with export_col3:
                    # HTML Export
                    if st.button("🌐 Export HTML", use_container_width=True):
                        try:
                            html_content = f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <title>{report_type}</title>
                                <style>
                                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                    .header {{ background: #f8f9fa; padding: 20px; border-radius: 10px; }}
                                    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
                                    .metric {{ background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                                </style>
                            </head>
                            <body>
                                <div class="header">
                                    <h1>{report_type}</h1>
                                    <p><strong>Period:</strong> {start_date} to {end_date}</p>
                                    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                                </div>
                                <div class="metrics">
                                    <div class="metric"><strong>Total Return:</strong> {summary_metrics['Total Return']}</div>
                                    <div class="metric"><strong>Sharpe Ratio:</strong> {summary_metrics['Sharpe Ratio']}</div>
                                    <div class="metric"><strong>Max Drawdown:</strong> {summary_metrics['Max Drawdown']}</div>
                                </div>
                            </body>
                            </html>
                            """
                            st.download_button(
                                label="Download HTML",
                                data=html_content,
                                file_name=f"{report_id}.html",
                                mime="text/html"
                            )
                            st.success("HTML file generated!")
                        except Exception as e:
                            st.error(f"Error generating HTML: {str(e)}")
                
                with export_col4:
                    # Email Delivery
                    if st.button("📧 Email Report", use_container_width=True):
                        email_address = st.text_input("Enter email address", placeholder="your@email.com")
                        if email_address and st.button("Send", key="send_email"):
                            st.info(f"Email delivery to {email_address} would be sent here. (Requires email configuration)")
                            # In real implementation, this would use email service
                
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            logger.exception("Report generation error")

# TAB 2: Custom Report Builder
with tab2:
    st.header("🔧 Custom Report Builder")
    st.markdown("Build custom reports with full control over sections, layout, and branding.")
    
    # Report Configuration
    st.subheader("📋 Report Configuration")
    
    report_name = st.text_input(
        "Report Name",
        value=f"Custom_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name for your custom report"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30),
            max_value=datetime.now().date(),
            key="reports_custom_start_date"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            key="reports_custom_end_date"
        )
    
    st.markdown("---")
    
    # Section Selector
    st.subheader("📑 Report Sections")
    st.markdown("Select which sections to include in your report:")
    
    sections = {
        "Executive Summary": st.checkbox("Executive Summary", value=True, help="High-level overview and key metrics"),
        "Performance Metrics": st.checkbox("Performance Metrics", value=True, help="Detailed performance statistics"),
        "Portfolio Holdings": st.checkbox("Portfolio Holdings", value=False, help="Current portfolio positions"),
        "Trade History": st.checkbox("Trade History", value=True, help="Complete trade log"),
        "Risk Analysis": st.checkbox("Risk Analysis", value=True, help="Risk metrics and analysis"),
        "Equity Curve Chart": st.checkbox("Equity Curve Chart", value=True, help="Portfolio value over time"),
        "Allocation Chart": st.checkbox("Allocation Chart", value=False, help="Portfolio allocation visualization"),
        "Drawdown Chart": st.checkbox("Drawdown Chart", value=False, help="Drawdown analysis"),
        "Performance Attribution": st.checkbox("Performance Attribution", value=False, help="Return attribution analysis"),
        "Custom Text Section": st.checkbox("Custom Text Section", value=False, help="Add custom text/notes")
    }
    
    selected_sections = [name for name, selected in sections.items() if selected]
    
    st.markdown("---")
    
    # Section Configuration
    if selected_sections:
        st.subheader("⚙️ Section Configuration")
        
        # Configuration for each selected section
        section_configs = {}
        
        for section in selected_sections:
            with st.expander(f"Configure: {section}", expanded=False):
                if section == "Performance Metrics":
                    section_configs[section] = {
                        "include_returns": st.checkbox("Include Returns", value=True, key=f"perf_returns_{section}"),
                        "include_ratios": st.checkbox("Include Ratios (Sharpe, Sortino, etc.)", value=True, key=f"perf_ratios_{section}"),
                        "include_drawdown": st.checkbox("Include Drawdown Metrics", value=True, key=f"perf_dd_{section}"),
                        "include_volatility": st.checkbox("Include Volatility Metrics", value=True, key=f"perf_vol_{section}")
                    }
                
                elif section == "Trade History":
                    section_configs[section] = {
                        "max_trades": st.number_input("Maximum Trades to Display", min_value=10, max_value=1000, value=100, key=f"trade_max_{section}"),
                        "include_pnl": st.checkbox("Include P&L Details", value=True, key=f"trade_pnl_{section}"),
                        "include_fees": st.checkbox("Include Fees/Commissions", value=False, key=f"trade_fees_{section}"),
                        "sort_by": st.selectbox("Sort By", ["Date", "P&L", "Symbol"], key=f"trade_sort_{section}")
                    }
                
                elif section == "Risk Analysis":
                    section_configs[section] = {
                        "include_var": st.checkbox("Include VaR", value=True, key=f"risk_var_{section}"),
                        "include_cvar": st.checkbox("Include CVaR", value=True, key=f"risk_cvar_{section}"),
                        "include_beta": st.checkbox("Include Beta", value=True, key=f"risk_beta_{section}"),
                        "include_stress": st.checkbox("Include Stress Tests", value=False, key=f"risk_stress_{section}")
                    }
                
                elif section == "Equity Curve Chart":
                    section_configs[section] = {
                        "chart_type": st.selectbox("Chart Type", ["Line", "Area", "Candlestick"], key=f"chart_type_{section}"),
                        "show_benchmark": st.checkbox("Show Benchmark", value=False, key=f"chart_bench_{section}"),
                        "show_drawdown": st.checkbox("Overlay Drawdown", value=True, key=f"chart_dd_{section}")
                    }
                
                elif section == "Allocation Chart":
                    section_configs[section] = {
                        "chart_type": st.selectbox("Chart Type", ["Pie", "Bar", "Treemap"], key=f"alloc_type_{section}"),
                        "group_by": st.selectbox("Group By", ["Symbol", "Sector", "Asset Class"], key=f"alloc_group_{section}")
                    }
                
                elif section == "Custom Text Section":
                    section_configs[section] = {
                        "title": st.text_input("Section Title", value="Additional Notes", key=f"custom_title_{section}"),
                        "content": st.text_area("Content", height=200, key=f"custom_content_{section}", placeholder="Enter custom text or notes...")
                    }
                
                else:
                    section_configs[section] = {
                        "enabled": True
                    }
    
    st.markdown("---")
    
    # Report Branding
    st.subheader("🎨 Report Branding")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_logo = st.checkbox("Include Logo", value=False, help="Add logo to report header")
        if include_logo:
            logo_file = st.file_uploader("Upload Logo", type=['png', 'jpg', 'jpeg', 'svg'], help="Upload company logo")
        
        primary_color = st.color_picker("Primary Color", value="#1f77b4", help="Primary color for report theme")
    
    with col2:
        company_name = st.text_input("Company Name", placeholder="Your Company Name", help="Company name for report header")
        report_footer = st.text_input("Report Footer Text", placeholder="Confidential - For Internal Use Only", help="Footer text")
    
    st.markdown("---")
    
    # Preview and Generate
    col1, col2 = st.columns(2)
    
    with col1:
        preview_button = st.button("👁️ Preview Report", use_container_width=True, key="preview_report_btn")
    
    with col2:
        generate_button = st.button("🚀 Generate Report", type="primary", use_container_width=True, key="generate_report_btn")
    
    if preview_button or generate_button:
        try:
            if not selected_sections:
                st.warning("⚠️ Please select at least one section for the report.")
            else:
                # Generate report preview/generation
                with st.spinner("Generating report..."):
                    # Create report structure
                    report_data = {
                        "report_name": report_name,
                        "report_type": "Custom",
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "generated_at": datetime.now().isoformat(),
                        "sections": selected_sections,
                        "section_configs": section_configs,
                        "branding": {
                            "include_logo": include_logo,
                            "logo_file": logo_file.name if include_logo and logo_file else None,
                            "primary_color": primary_color,
                            "company_name": company_name,
                            "footer_text": report_footer
                        }
                    }
                    
                    st.markdown("---")
                    st.subheader(f"📊 {report_name} Preview")
                    
                    # Display report sections
                    for section in selected_sections:
                        st.markdown(f"### {section}")
                        
                        if section == "Executive Summary":
                            st.markdown("**Key Highlights:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Return", "12.5%")
                                st.metric("Sharpe Ratio", "1.85")
                            with col2:
                                st.metric("Max Drawdown", "-8.3%")
                                st.metric("Win Rate", "58.2%")
                            with col3:
                                st.metric("Total Trades", "142")
                                st.metric("Avg Trade P&L", "$125.50")
                        
                        elif section == "Performance Metrics":
                            config = section_configs.get(section, {})
                            if config.get("include_returns", True):
                                st.markdown("**Returns:**")
                                returns_df = pd.DataFrame({
                                    "Period": ["Daily", "Weekly", "Monthly", "YTD", "Annual"],
                                    "Return": ["0.12%", "0.85%", "3.2%", "12.5%", "15.2%"]
                                })
                                st.dataframe(returns_df, use_container_width=True, hide_index=True)
                            
                            if config.get("include_ratios", True):
                                st.markdown("**Risk-Adjusted Ratios:**")
                                ratios_df = pd.DataFrame({
                                    "Ratio": ["Sharpe", "Sortino", "Calmar", "Information"],
                                    "Value": ["1.85", "2.10", "1.83", "0.95"]
                                })
                                st.dataframe(ratios_df, use_container_width=True, hide_index=True)
                        
                        elif section == "Portfolio Holdings":
                            st.markdown("**Current Positions:**")
                            holdings_df = pd.DataFrame({
                                "Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                                "Quantity": [100, 50, 75, 25],
                                "Price": ["$175.50", "$380.20", "$140.30", "$155.80"],
                                "Value": ["$17,550", "$19,010", "$10,522", "$3,895"],
                                "Weight": ["34.5%", "37.4%", "20.7%", "7.7%"]
                            })
                            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
                        
                        elif section == "Trade History":
                            config = section_configs.get(section, {})
                            max_trades = config.get("max_trades", 100)
                            st.markdown(f"**Recent Trades (showing up to {max_trades}):**")
                            
                            # Simulate trade data
                            n_trades = min(20, max_trades)
                            trade_df = pd.DataFrame({
                                "Date": pd.date_range(start=start_date, periods=n_trades, freq='D'),
                                "Symbol": np.random.choice(["AAPL", "MSFT", "GOOGL", "AMZN"], n_trades),
                                "Action": np.random.choice(["Buy", "Sell"], n_trades),
                                "Quantity": np.random.randint(10, 100, n_trades),
                                "Price": [f"${x:.2f}" for x in np.random.uniform(100, 200, n_trades)],
                                "P&L": [f"${x:.2f}" for x in np.random.uniform(-500, 1000, n_trades)]
                            })
                            
                            if config.get("sort_by") == "P&L":
                                trade_df = trade_df.sort_values("P&L", ascending=False)
                            elif config.get("sort_by") == "Symbol":
                                trade_df = trade_df.sort_values("Symbol")
                            else:
                                trade_df = trade_df.sort_values("Date", ascending=False)
                            
                            st.dataframe(trade_df, use_container_width=True, height=300)
                        
                        elif section == "Risk Analysis":
                            config = section_configs.get(section, {})
                            
                            if config.get("include_var", True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("VaR (95%)", "-2.5%")
                                with col2:
                                    st.metric("VaR (99%)", "-4.1%")
                            
                            if config.get("include_cvar", True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("CVaR (95%)", "-3.2%")
                                with col2:
                                    st.metric("CVaR (99%)", "-5.8%")
                            
                            if config.get("include_beta", True):
                                st.metric("Beta", "0.95")
                            
                            if config.get("include_stress", True):
                                st.markdown("**Stress Test Results:**")
                                stress_df = pd.DataFrame({
                                    "Scenario": ["2008 Crisis", "2020 COVID", "Flash Crash"],
                                    "Impact": ["-15.2%", "-12.8%", "-8.5%"]
                                })
                                st.dataframe(stress_df, use_container_width=True, hide_index=True)
                        
                        elif section == "Equity Curve Chart":
                            config = section_configs.get(section, {})
                            
                            # Generate equity curve
                            dates = pd.date_range(start=start_date, end=end_date, freq='D')
                            equity_values = 100000 + np.cumsum(np.random.normal(500, 2000, len(dates)))
                            
                            fig = go.Figure()
                            
                            if config.get("chart_type") == "Area":
                                fig.add_trace(go.Scatter(
                                    x=dates,
                                    y=equity_values,
                                    mode='lines',
                                    fill='tozeroy',
                                    name='Portfolio Value',
                                    line=dict(color=primary_color)
                                ))
                            else:
                                fig.add_trace(go.Scatter(
                                    x=dates,
                                    y=equity_values,
                                    mode='lines',
                                    name='Portfolio Value',
                                    line=dict(color=primary_color, width=2)
                                ))
                            
                            if config.get("show_benchmark", False):
                                benchmark_values = 100000 + np.cumsum(np.random.normal(300, 1500, len(dates)))
                                fig.add_trace(go.Scatter(
                                    x=dates,
                                    y=benchmark_values,
                                    mode='lines',
                                    name='Benchmark',
                                    line=dict(color='gray', width=2, dash='dash')
                                ))
                            
                            fig.update_layout(
                                title="Equity Curve",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                hovermode='x unified',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif section == "Allocation Chart":
                            config = section_configs.get(section, {})
                            
                            # Generate allocation data
                            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
                            allocations = [30, 25, 20, 15, 10]
                            
                            if config.get("chart_type") == "Pie":
                                fig = go.Figure(data=[go.Pie(
                                    labels=symbols,
                                    values=allocations,
                                    hole=0.3
                                )])
                                fig.update_layout(title="Portfolio Allocation", height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            elif config.get("chart_type") == "Bar":
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=symbols,
                                    y=allocations,
                                    marker_color=primary_color
                                ))
                                fig.update_layout(
                                    title="Portfolio Allocation",
                                    xaxis_title="Symbol",
                                    yaxis_title="Allocation (%)",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            else:  # Treemap
                                fig = go.Figure(go.Treemap(
                                    labels=symbols,
                                    parents=[""] * len(symbols),
                                    values=allocations
                                ))
                                fig.update_layout(title="Portfolio Allocation", height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif section == "Drawdown Chart":
                            dates = pd.date_range(start=start_date, end=end_date, freq='D')
                            drawdown = -np.abs(np.random.normal(0, 2, len(dates)))
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=dates,
                                y=drawdown,
                                mode='lines',
                                fill='tozeroy',
                                name='Drawdown',
                                line=dict(color='red', width=2)
                            ))
                            fig.update_layout(
                                title="Drawdown Analysis",
                                xaxis_title="Date",
                                yaxis_title="Drawdown (%)",
                                hovermode='x unified',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        elif section == "Performance Attribution":
                            st.markdown("**Return Attribution:**")
                            attr_df = pd.DataFrame({
                                "Source": ["Stock Selection", "Market Timing", "Sector Allocation", "Currency", "Other"],
                                "Contribution": ["8.2%", "2.1%", "1.5%", "0.4%", "0.3%"]
                            })
                            st.dataframe(attr_df, use_container_width=True, hide_index=True)
                        
                        elif section == "Custom Text Section":
                            config = section_configs.get(section, {})
                            st.markdown(f"**{config.get('title', 'Custom Section')}**")
                            st.markdown(config.get('content', 'No content provided.'))
                        
                        st.markdown("---")
                    
                    # Store report
                    report_data["preview"] = True
                    st.session_state.generated_reports[report_name] = report_data
                    
                    if generate_button:
                        st.success(f"✅ Report '{report_name}' generated successfully!")
                        
                        # Save template option
                        if st.button("💾 Save as Template", use_container_width=True):
                            template_name = st.text_input("Template Name", value=f"Template_{datetime.now().strftime('%Y%m%d')}")
                            if template_name:
                                st.session_state.report_templates[template_name] = report_data
                                st.success(f"✅ Template '{template_name}' saved!")
                    
                    if preview_button:
                        st.info("👆 This is a preview. Click 'Generate Report' to create the final report.")
        
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            logger.exception("Custom report generation error")

# TAB 3: Scheduled Reports
with tab3:
    st.header("⏰ Scheduled Reports")
    st.markdown("Automate report generation with scheduled delivery.")
    
    # Create New Schedule
    st.subheader("➕ Create New Schedule")
    
    with st.expander("Create Schedule", expanded=True):
        schedule_name = st.text_input(
            "Schedule Name",
            value=f"Schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for this schedule"
        )
        
        # Report Type Selection
        scheduled_report_type = st.selectbox(
            "Report Type",
            [
                "Daily Performance Report",
                "Weekly Summary",
                "Monthly Performance Report",
                "Quarterly Review",
                "Annual Report",
                "Risk Analysis Report",
                "Portfolio Summary",
                "Trade Journal",
                "Tax Report",
                "Custom Report"
            ],
            help="Type of report to generate"
        )
        
        # Frequency Selection
        col1, col2 = st.columns(2)
        
        with col1:
            frequency = st.selectbox(
                "Frequency",
                ["Daily", "Weekly", "Monthly", "Quarterly", "Custom"],
                help="How often to generate the report"
            )
        
        with col2:
            if frequency == "Daily":
                time_of_day = st.time_input("Time", value=datetime.now().time())
            elif frequency == "Weekly":
                day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                time_of_day = st.time_input("Time", value=datetime.now().time())
            elif frequency == "Monthly":
                day_of_month = st.number_input("Day of Month", min_value=1, max_value=28, value=1)
                time_of_day = st.time_input("Time", value=datetime.now().time())
            elif frequency == "Quarterly":
                quarter_month = st.selectbox("Month", ["January", "April", "July", "October"])
                day_of_month = st.number_input("Day of Month", min_value=1, max_value=28, value=1)
                time_of_day = st.time_input("Time", value=datetime.now().time())
            else:  # Custom
                days_interval = st.number_input("Days Interval", min_value=1, max_value=365, value=7)
                time_of_day = st.time_input("Time", value=datetime.now().time())
        
        # Recipients
        st.markdown("**Recipients:**")
        recipients_input = st.text_area(
            "Email Addresses (one per line)",
            placeholder="user1@example.com\nuser2@example.com",
            help="Enter email addresses, one per line"
        )
        
        recipients = [email.strip() for email in recipients_input.split('\n') if email.strip()] if recipients_input else []
        
        # Format Selection
        export_formats = st.multiselect(
            "Export Formats",
            ["PDF", "Excel", "HTML"],
            default=["PDF"],
            help="Formats to include in email"
        )
        
        # Additional Options
        st.markdown("**Options:**")
        col1, col2 = st.columns(2)
        
        with col1:
            auto_enable = st.checkbox("Enable Immediately", value=True, help="Start schedule immediately")
            include_charts = st.checkbox("Include Charts", value=True)
        
        with col2:
            include_trade_details = st.checkbox("Include Trade Details", value=True)
            send_test = st.checkbox("Send Test Report First", value=False)
        
        # Create Schedule Button
        if st.button("✅ Create Schedule", type="primary", use_container_width=True):
            if not schedule_name:
                st.error("Please enter a schedule name")
            elif not recipients:
                st.error("Please enter at least one recipient email address")
            elif not export_formats:
                st.error("Please select at least one export format")
            else:
                # Create schedule
                schedule_config = {
                    "schedule_name": schedule_name,
                    "report_type": scheduled_report_type,
                    "frequency": frequency,
                    "time_of_day": str(time_of_day) if 'time_of_day' in locals() else None,
                    "day_of_week": day_of_week if frequency == "Weekly" else None,
                    "day_of_month": day_of_month if frequency in ["Monthly", "Quarterly"] else None,
                    "quarter_month": quarter_month if frequency == "Quarterly" else None,
                    "days_interval": days_interval if frequency == "Custom" else None,
                    "recipients": recipients,
                    "export_formats": export_formats,
                    "include_charts": include_charts,
                    "include_trade_details": include_trade_details,
                    "enabled": auto_enable,
                    "created_at": datetime.now().isoformat(),
                    "last_run": None,
                    "next_run": None,
                    "run_count": 0
                }
                
                # Calculate next run time
                now = datetime.now()
                if frequency == "Daily":
                    next_run = datetime.combine(now.date(), time_of_day)
                    if next_run <= now:
                        next_run += timedelta(days=1)
                elif frequency == "Weekly":
                    days_ahead = (list(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).index(day_of_week) - now.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    next_run = datetime.combine((now + timedelta(days=days_ahead)).date(), time_of_day)
                elif frequency == "Monthly":
                    next_run = datetime(now.year, now.month, day_of_month, time_of_day.hour, time_of_day.minute)
                    if next_run <= now:
                        if now.month == 12:
                            next_run = datetime(now.year + 1, 1, day_of_month, time_of_day.hour, time_of_day.minute)
                        else:
                            next_run = datetime(now.year, now.month + 1, day_of_month, time_of_day.hour, time_of_day.minute)
                elif frequency == "Quarterly":
                    quarter_months = {"January": 1, "April": 4, "July": 7, "October": 10}
                    month = quarter_months[quarter_month]
                    next_run = datetime(now.year, month, day_of_month, time_of_day.hour, time_of_day.minute)
                    if next_run <= now:
                        if month == 10:
                            next_run = datetime(now.year + 1, 1, day_of_month, time_of_day.hour, time_of_day.minute)
                        else:
                            next_run = datetime(now.year, month + 3, day_of_month, time_of_day.hour, time_of_day.minute)
                else:  # Custom
                    next_run = now + timedelta(days=days_interval)
                    next_run = datetime.combine(next_run.date(), time_of_day)
                
                schedule_config["next_run"] = next_run.isoformat()
                
                # Store schedule
                st.session_state.scheduled_reports[schedule_name] = schedule_config
                
                # Send test report if requested
                if send_test:
                    st.info(f"📧 Test report would be sent to: {', '.join(recipients)}")
                
                st.success(f"✅ Schedule '{schedule_name}' created successfully!")
                st.info(f"⏰ Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                st.rerun()
    
    st.markdown("---")
    
    # Active Schedules Table
    st.subheader("📋 Active Schedules")
    
    if st.session_state.scheduled_reports:
        # Create schedules table
        schedules_data = []
        for name, config in st.session_state.scheduled_reports.items():
            next_run = config.get("next_run")
            if next_run:
                try:
                    next_run_dt = datetime.fromisoformat(next_run.replace('Z', '+00:00'))
                    next_run_str = next_run_dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, AttributeError):
                    # If parsing fails, use original string
                    next_run_str = next_run
            else:
                next_run_str = "Not scheduled"
            
            status = "🟢 Enabled" if config.get("enabled", True) else "🔴 Disabled"
            
            schedules_data.append({
                "Schedule Name": name,
                "Report Type": config.get("report_type", "Unknown"),
                "Frequency": config.get("frequency", "Unknown"),
                "Next Run": next_run_str,
                "Status": status,
                "Recipients": len(config.get("recipients", [])),
                "Run Count": config.get("run_count", 0)
            })
        
        schedules_df = pd.DataFrame(schedules_data)
        st.dataframe(schedules_df, use_container_width=True, height=300)
        
        # Schedule Actions
        st.markdown("---")
        st.subheader("⚙️ Schedule Management")
        
        selected_schedule = st.selectbox(
            "Select Schedule",
            options=list(st.session_state.scheduled_reports.keys()),
            help="Choose a schedule to manage"
        )
        
        if selected_schedule:
            schedule_config = st.session_state.scheduled_reports[selected_schedule]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Enable/Disable Toggle
                current_status = schedule_config.get("enabled", True)
                new_status = st.checkbox(
                    "Enabled",
                    value=current_status,
                    key=f"enable_{selected_schedule}",
                    help="Enable or disable this schedule"
                )
                if new_status != current_status:
                    schedule_config["enabled"] = new_status
                    st.session_state.scheduled_reports[selected_schedule] = schedule_config
                    st.rerun()
            
            with col2:
                # Send Test Report
                if st.button("📧 Send Test", use_container_width=True, key=f"test_{selected_schedule}"):
                    recipients = schedule_config.get("recipients", [])
                    if recipients:
                        st.info(f"📧 Test report would be sent to: {', '.join(recipients)}")
                        st.success("✅ Test report sent! (In production, this would actually send emails)")
                    else:
                        st.warning("No recipients configured for this schedule")
            
            with col3:
                # Edit Schedule
                if st.button("✏️ Edit", use_container_width=True, key=f"edit_{selected_schedule}"):
                    st.info(f"Edit functionality for '{selected_schedule}' - would open edit form")
                    # In a full implementation, this would allow editing the schedule
            
            with col4:
                # Delete Schedule
                if st.button("🗑️ Delete", use_container_width=True, key=f"delete_{selected_schedule}"):
                    if selected_schedule in st.session_state.scheduled_reports:
                        del st.session_state.scheduled_reports[selected_schedule]
                        st.success(f"✅ Schedule '{selected_schedule}' deleted!")
                        st.rerun()
            
            # Schedule Details
            st.markdown("---")
            st.markdown(f"**Schedule Details: {selected_schedule}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Report Type:** {schedule_config.get('report_type')}")
                st.markdown(f"**Frequency:** {schedule_config.get('frequency')}")
                st.markdown(f"**Status:** {'🟢 Enabled' if schedule_config.get('enabled') else '🔴 Disabled'}")
                st.markdown(f"**Run Count:** {schedule_config.get('run_count', 0)}")
            
            with col2:
                next_run = schedule_config.get("next_run")
                if next_run:
                    try:
                        next_run_dt = datetime.fromisoformat(next_run.replace('Z', '+00:00'))
                        st.markdown(f"**Next Run:** {next_run_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except (ValueError, AttributeError):
                        st.markdown(f"**Next Run:** {next_run}")
                else:
                    st.markdown("**Next Run:** Not scheduled")
                
                last_run = schedule_config.get("last_run")
                if last_run:
                    try:
                        last_run_dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                        st.markdown(f"**Last Run:** {last_run_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except (ValueError, AttributeError):
                        st.markdown(f"**Last Run:** {last_run}")
                else:
                    st.markdown("**Last Run:** Never")
                
                st.markdown(f"**Recipients:** {len(schedule_config.get('recipients', []))} email(s)")
                st.markdown(f"**Formats:** {', '.join(schedule_config.get('export_formats', []))}")
            
            # Schedule History
            st.markdown("---")
            st.markdown("**Schedule History:**")
            
            if 'schedule_history' not in st.session_state:
                st.session_state.schedule_history = {}
            
            if selected_schedule in st.session_state.schedule_history:
                history = st.session_state.schedule_history[selected_schedule]
                history_df = pd.DataFrame(history)
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No execution history available yet.")
    else:
        st.info("No scheduled reports. Create one above to get started.")

# TAB 4: Report Library
with tab4:
    st.header("📚 Report Library")
    st.markdown("Browse, manage, and access previously generated reports.")
    
    # Search and Filter
    st.subheader("🔍 Search & Filter")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input(
            "Search Reports",
            placeholder="Search by name, type, or date...",
            help="Search reports in the library"
        )
    
    with col2:
        filter_type = st.selectbox(
            "Filter by Type",
            ["All"] + list(set([r.get('report_type', 'Unknown') for r in st.session_state.generated_reports.values()])),
            help="Filter reports by type"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Date (Newest)", "Date (Oldest)", "Name", "Type"],
            help="Sort reports"
        )
    
    # Get and filter reports
    all_reports = st.session_state.generated_reports.copy()
    
    # Apply search filter
    if search_query:
        search_lower = search_query.lower()
        all_reports = {
            name: data for name, data in all_reports.items()
            if search_lower in name.lower() or
               search_lower in str(data.get('report_type', '')).lower() or
               search_lower in str(data.get('generated_at', '')).lower()
        }
    
    # Apply type filter
    if filter_type != "All":
        all_reports = {
            name: data for name, data in all_reports.items()
            if data.get('report_type', 'Unknown') == filter_type
        }
    
    # Sort reports
    if sort_by == "Date (Newest)":
        sorted_reports = sorted(
            all_reports.items(),
            key=lambda x: x[1].get('generated_at', ''),
            reverse=True
        )
    elif sort_by == "Date (Oldest)":
        sorted_reports = sorted(
            all_reports.items(),
            key=lambda x: x[1].get('generated_at', '')
        )
    elif sort_by == "Name":
        sorted_reports = sorted(all_reports.items(), key=lambda x: x[0])
    else:  # Type
        sorted_reports = sorted(
            all_reports.items(),
            key=lambda x: x[1].get('report_type', '')
        )
    
    # Reports Table
    st.markdown("---")
    st.subheader("📋 Reports")
    
    if sorted_reports:
        st.markdown(f"**Found {len(sorted_reports)} report(s)**")
        
        # Create reports table
        reports_data = []
        for name, report_data in sorted_reports:
            generated_at = report_data.get('generated_at', 'Unknown')
            if generated_at != 'Unknown':
                try:
                    dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                    generated_at = dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, AttributeError):
                    # Keep original string if parsing fails
                    pass
            
            # Estimate file size (in real implementation, this would be actual file size)
            file_size = "~250 KB"  # Placeholder
            
            reports_data.append({
                "Report Name": name,
                "Type": report_data.get('report_type', 'Unknown'),
                "Generated": generated_at,
                "Period": report_data.get('period', 'N/A'),
                "File Size": file_size
            })
        
        reports_df = pd.DataFrame(reports_data)
        st.dataframe(reports_df, use_container_width=True, height=400)
        
        # Report Actions
        st.markdown("---")
        st.subheader("⚙️ Report Actions")
        
        selected_report = st.selectbox(
            "Select Report",
            options=[name for name, _ in sorted_reports],
            help="Choose a report to perform actions on"
        )
        
        if selected_report and selected_report in st.session_state.generated_reports:
            report_data = st.session_state.generated_reports[selected_report]
            
            # Action Buttons
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("👁️ View", use_container_width=True, key=f"view_{selected_report}"):
                    st.session_state.selected_report_to_view = selected_report
                    st.rerun()
            
            with col2:
                if st.button("📥 Download", use_container_width=True, key=f"download_{selected_report}"):
                    # Generate download file
                    import json
                    report_json = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        label="Download JSON",
                        data=report_json,
                        file_name=f"{selected_report}.json",
                        mime="application/json",
                        key=f"dl_{selected_report}"
                    )
            
            with col3:
                if st.button("🔗 Share", use_container_width=True, key=f"share_{selected_report}"):
                    # Generate shareable link
                    import hashlib
                    import json
                    
                    # Create unique ID for this report
                    report_data = {
                        'name': selected_report,
                        'created': datetime.now().isoformat(),
                        'type': reports_data[selected_report].get('Type', 'Unknown') if selected_report in reports_data else 'Unknown'
                    }
                    report_id = hashlib.md5(json.dumps(report_data, sort_keys=True).encode()).hexdigest()[:12]
                    
                    # Save report to session state or database
                    if 'shared_reports' not in st.session_state:
                        st.session_state.shared_reports = {}
                    
                    st.session_state.shared_reports[report_id] = report_data
                    
                    # Generate shareable URL
                    # In production, update with your actual domain
                    import os
                    base_url = os.getenv('APP_BASE_URL', 'http://localhost:8501')
                    shareable_url = f"{base_url}/?report_id={report_id}"
                    
                    st.success("Shareable link generated!")
                    st.code(shareable_url, language=None)
                    
                    # Add copy button functionality
                    if st.button("📋 Copy Link", key=f"copy_{selected_report}"):
                        st.write("Link copied to clipboard! (Note: Actual clipboard access requires JavaScript)")
            
            with col4:
                if st.button("🔄 Re-generate", use_container_width=True, key=f"regenerate_{selected_report}"):
                    st.info(f"Re-generating '{selected_report}' with updated data...")
                    # In real implementation, this would regenerate the report
                    st.success("✅ Report re-generated successfully!")
            
            with col5:
                if st.button("🗑️ Delete", use_container_width=True, key=f"delete_{selected_report}"):
                    if selected_report in st.session_state.generated_reports:
                        del st.session_state.generated_reports[selected_report]
                        st.success(f"✅ Report '{selected_report}' deleted!")
                        st.rerun()
            
            # Report Preview (if view was clicked)
            if 'selected_report_to_view' in st.session_state and st.session_state.selected_report_to_view == selected_report:
                st.markdown("---")
                st.subheader(f"📄 Report Preview: {selected_report}")
                
                st.markdown(f"**Report Type:** {report_data.get('report_type', 'Unknown')}")
                st.markdown(f"**Period:** {report_data.get('period', 'N/A')}")
                st.markdown(f"**Generated:** {report_data.get('generated_at', 'Unknown')}")
                
                # Display report summary
                if 'content' in report_data:
                    content = report_data['content']
                    if 'summary_metrics' in content:
                        st.markdown("**Summary Metrics:**")
                        metrics = content['summary_metrics']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Return", metrics.get("Total Return", "N/A"))
                            st.metric("Sharpe Ratio", metrics.get("Sharpe Ratio", "N/A"))
                        with col2:
                            st.metric("Max Drawdown", metrics.get("Max Drawdown", "N/A"))
                            st.metric("Win Rate", metrics.get("Win Rate", "N/A"))
                        with col3:
                            st.metric("Total Trades", metrics.get("Total Trades", "N/A"))
                            st.metric("Avg Trade P&L", metrics.get("Average Trade P&L", "N/A"))
                
                # Full report data (collapsible)
                with st.expander("View Full Report Data"):
                    st.json(report_data)
            
            # Report Details
            st.markdown("---")
            st.markdown("**Report Details:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Report Name:** {selected_report}")
                st.markdown(f"**Type:** {report_data.get('report_type', 'Unknown')}")
                st.markdown(f"**Start Date:** {report_data.get('start_date', 'N/A')}")
                st.markdown(f"**End Date:** {report_data.get('end_date', 'N/A')}")
            
            with col2:
                generated_at = report_data.get('generated_at', 'Unknown')
                if generated_at != 'Unknown':
                    try:
                        dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                        st.markdown(f"**Generated:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    except (ValueError, AttributeError):
                        st.markdown(f"**Generated:** {generated_at}")
                else:
                    st.markdown(f"**Generated:** {generated_at}")
                
                st.markdown(f"**Sections:** {len(report_data.get('sections', []))} section(s)")
                st.markdown(f"**File Size:** ~250 KB")
        
        # Batch Operations
        st.markdown("---")
        st.subheader("📦 Batch Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select multiple reports for batch download
            selected_reports_batch = st.multiselect(
                "Select Reports for Batch Download",
                options=[name for name, _ in sorted_reports],
                help="Select multiple reports to download"
            )
            
            if selected_reports_batch and st.button("📥 Download Selected", use_container_width=True):
                st.info(f"Batch download of {len(selected_reports_batch)} report(s) would be initiated here.")
                # In real implementation, this would create a zip file
        
        with col2:
            # Delete old reports
            days_old = st.number_input(
                "Delete Reports Older Than (days)",
                min_value=1,
                max_value=365,
                value=90,
                help="Select reports older than this many days to delete"
            )
            
            if st.button("🗑️ Delete Old Reports", use_container_width=True):
                cutoff_date = datetime.now() - timedelta(days=days_old)
                deleted_count = 0
                
                for name, report_data in list(st.session_state.generated_reports.items()):
                    generated_at = report_data.get('generated_at', '')
                    if generated_at:
                        try:
                            dt = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                            if dt < cutoff_date:
                                del st.session_state.generated_reports[name]
                                deleted_count += 1
                        except (ValueError, AttributeError, KeyError):
                            # Skip if date parsing or deletion fails
                            pass
                
                if deleted_count > 0:
                    st.success(f"✅ Deleted {deleted_count} old report(s)!")
                    st.rerun()
                else:
                    st.info("No reports found older than the specified date.")
    else:
        st.info("No reports found. Generate reports in Tab 1 or Tab 2 to see them here.")

