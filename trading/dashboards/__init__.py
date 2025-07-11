"""Dashboard components for trading visualization."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import os
import importlib
import inspect
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# --- Plugin System ---
class DashboardPlugin(ABC):
    """Abstract base class for dashboard plugins."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get plugin description."""
        pass
    
    @abstractmethod
    def render(self, data: Dict[str, Any]) -> None:
        """Render the dashboard component."""
        pass
    
    @abstractmethod
    def get_required_data_keys(self) -> List[str]:
        """Get list of required data keys for this plugin."""
        pass

class PluginManager:
    """Manages dashboard plugin discovery and loading."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self.plugins: Dict[str, DashboardPlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self._discover_plugins()
    
    def _discover_plugins(self):
        """Discover and load dashboard plugins."""
        try:
            # Look for plugins in the dashboards directory
            dashboards_dir = Path(__file__).parent
            plugin_dirs = [
                dashboards_dir / "plugins",
                dashboards_dir / "layouts",
                dashboards_dir / "components"
            ]
            
            for plugin_dir in plugin_dirs:
                if plugin_dir.exists():
                    self._load_plugins_from_directory(plugin_dir)
            
            # Look for plugins in the main dashboards directory
            self._load_plugins_from_directory(dashboards_dir)
            
            logger.info(f"Discovered {len(self.plugins)} dashboard plugins")
            
        except Exception as e:
            logger.error(f"Error discovering plugins: {e}")
    
    def _load_plugins_from_directory(self, directory: Path):
        """Load plugins from a specific directory."""
        try:
            for file_path in directory.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                
                try:
                    # Import the module
                    module_name = f"trading.dashboards.{file_path.stem}"
                    if directory.name in ["plugins", "layouts", "components"]:
                        module_name = f"trading.dashboards.{directory.name}.{file_path.stem}"
                    
                    module = importlib.import_module(module_name)
                    
                    # Look for classes that inherit from DashboardPlugin
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, DashboardPlugin) and 
                            obj != DashboardPlugin):
                            
                            try:
                                plugin_instance = obj()
                                plugin_name = plugin_instance.get_name()
                                self.plugins[plugin_name] = plugin_instance
                                
                                # Load plugin configuration if available
                                config_file = file_path.with_suffix('.json')
                                if config_file.exists():
                                    import json
                                    with open(config_file, 'r') as f:
                                        self.plugin_configs[plugin_name] = json.load(f)
                                
                                logger.info(f"Loaded plugin: {plugin_name}")
                                
                            except Exception as e:
                                logger.warning(f"Error instantiating plugin {name}: {e}")
                                continue
                
                except Exception as e:
                    logger.warning(f"Error loading plugin from {file_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error loading plugins from directory {directory}: {e}")
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugin names."""
        return list(self.plugins.keys())
    
    def get_plugin(self, plugin_name: str) -> Optional[DashboardPlugin]:
        """Get a specific plugin by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        return self.plugin_configs.get(plugin_name, {})
    
    def render_plugin(self, plugin_name: str, data: Dict[str, Any]) -> bool:
        """Render a specific plugin.
        
        Args:
            plugin_name: Name of the plugin to render
            data: Data to pass to the plugin
            
        Returns:
            True if successful, False otherwise
        """
        try:
            plugin = self.get_plugin(plugin_name)
            if plugin is None:
                logger.error(f"Plugin '{plugin_name}' not found")
                return False
            
            # Check if required data is available
            required_keys = plugin.get_required_data_keys()
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                logger.warning(f"Missing required data keys for plugin '{plugin_name}': {missing_keys}")
                st.warning(f"Missing required data: {', '.join(missing_keys)}")
                return False
            
            # Render the plugin
            plugin.render(data)
            return True
            
        except Exception as e:
            logger.error(f"Error rendering plugin '{plugin_name}': {e}")
            st.error(f"Error rendering plugin: {e}")
            return False
    
    def render_all_plugins(self, data: Dict[str, Any]) -> None:
        """Render all available plugins.
        
        Args:
            data: Data to pass to all plugins
        """
        for plugin_name in self.get_available_plugins():
            try:
                st.subheader(plugin_name)
                self.render_plugin(plugin_name, data)
                st.divider()
            except Exception as e:
                logger.error(f"Error rendering plugin '{plugin_name}': {e}")

# --- Streamlit Layout Manager ---
class StreamlitLayoutManager:
    """Manages Streamlit layout components."""
    
    def __init__(self):
        """Initialize layout manager."""
        self.layouts: Dict[str, callable] = {}
        self._discover_layouts()
    
    def _discover_layouts(self):
        """Discover Streamlit layout functions."""
        try:
            # Look for layout files
            layouts_dir = Path(__file__).parent / "layouts"
            if layouts_dir.exists():
                for file_path in layouts_dir.glob("*.py"):
                    if file_path.name.startswith("__"):
                        continue
                    
                    try:
                        # Import the module
                        module_name = f"trading.dashboards.layouts.{file_path.stem}"
                        module = importlib.import_module(module_name)
                        
                        # Look for functions that might be layouts
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isfunction(obj) and 
                                not name.startswith("_") and
                                "render" in name.lower()):
                                
                                self.layouts[name] = obj
                                logger.info(f"Discovered layout: {name}")
                    
                    except Exception as e:
                        logger.warning(f"Error loading layout from {file_path}: {e}")
                        continue
            
            logger.info(f"Discovered {len(self.layouts)} Streamlit layouts")
            
        except Exception as e:
            logger.error(f"Error discovering layouts: {e}")
    
    def get_available_layouts(self) -> List[str]:
        """Get list of available layout names."""
        return list(self.layouts.keys())
    
    def render_layout(self, layout_name: str, **kwargs) -> bool:
        """Render a specific layout.
        
        Args:
            layout_name: Name of the layout to render
            **kwargs: Arguments to pass to the layout function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            layout_func = self.layouts.get(layout_name)
            if layout_func is None:
                logger.error(f"Layout '{layout_name}' not found")
                return False
            
            # Render the layout
            layout_func(**kwargs)
            return True
            
        except Exception as e:
            logger.error(f"Error rendering layout '{layout_name}': {e}")
            st.error(f"Error rendering layout: {e}")
            return False

# --- Global Plugin and Layout Managers ---
_plugin_manager = None
_layout_manager = None

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

def get_layout_manager() -> StreamlitLayoutManager:
    """Get the global layout manager instance."""
    global _layout_manager
    if _layout_manager is None:
        _layout_manager = StreamlitLayoutManager()
    return _layout_manager

class TradingDashboard:
    """Main trading dashboard component."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize trading dashboard."""
        self.config = config or {}
        self.plugin_manager = get_plugin_manager()
        self.layout_manager = get_layout_manager()
    
    def render_portfolio_overview(self, portfolio_data: pd.DataFrame):
        """Render portfolio overview chart."""
        try:
            if portfolio_data.empty:
                st.warning("No portfolio data available")

            fig = go.Figure()
            
            # Portfolio value over time
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['total_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering portfolio overview: {e}")
            st.error("Error rendering portfolio chart")
    
    def render_returns_distribution(self, returns: pd.Series):
        """Render returns distribution chart."""
        try:
            if returns.empty:
                st.warning("No returns data available")

            fig = px.histogram(
                returns,
                title='Returns Distribution',
                nbins=50,
                color_discrete_sequence=['blue']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering returns distribution: {e}")
            st.error("Error rendering returns chart")
    
    def render_drawdown_chart(self, portfolio_data: pd.DataFrame):
        """Render drawdown chart."""
        try:
            if portfolio_data.empty:
                st.warning("No portfolio data available")

            # Calculate drawdown
            peak = portfolio_data['total_value'].expanding().max()
            drawdown = (portfolio_data['total_value'] - peak) / peak * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_data.index,
                y=drawdown,
                mode='lines',
                name='Drawdown (%)',
                line=dict(color='red'),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering drawdown chart: {e}")
            st.error("Error rendering drawdown chart")
    
    def render_asset_allocation(self, allocation_data: Dict[str, float]):
        """Render asset allocation pie chart."""
        try:
            if not allocation_data:
                st.warning("No allocation data available")

            fig = go.Figure(data=[go.Pie(
                labels=list(allocation_data.keys()),
                values=list(allocation_data.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title='Asset Allocation',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering asset allocation: {e}")
            st.error("Error rendering allocation chart")
    
    def render_performance_metrics(self, metrics: Dict[str, float]):
        """Render performance metrics table."""
        try:
            if not metrics:
                st.warning("No metrics available")

            # Create metrics dataframe
            metrics_df = pd.DataFrame([
                {'Metric': k, 'Value': f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in metrics.items()
            ])
            
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering performance metrics: {e}")
            st.error("Error rendering metrics table")
    
    def render_with_plugins(self, data: Dict[str, Any], plugin_names: Optional[List[str]] = None):
        """Render dashboard with plugins.
        
        Args:
            data: Data to pass to plugins
            plugin_names: Optional list of specific plugins to render
        """
        if plugin_names is None:
            # Render all plugins
            self.plugin_manager.render_all_plugins(data)
        else:
            # Render specific plugins
            for plugin_name in plugin_names:
                self.plugin_manager.render_plugin(plugin_name, data)
    
    def render_with_layout(self, layout_name: str, **kwargs):
        """Render dashboard with a specific layout.
        
        Args:
            layout_name: Name of the layout to use
            **kwargs: Arguments to pass to the layout
        """
        return self.layout_manager.render_layout(layout_name, **kwargs)
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugins."""
        return self.plugin_manager.get_available_plugins()
    
    def get_available_layouts(self) -> List[str]:
        """Get list of available layouts."""
        return self.layout_manager.get_available_layouts()

class StrategyDashboard:
    """Strategy-specific dashboard component."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy dashboard."""
        self.config = config or {}
        self.plugin_manager = get_plugin_manager()
    
    def render_strategy_performance(self, strategy_data: pd.DataFrame):
        """Render strategy performance chart."""
        try:
            if strategy_data.empty:
                st.warning("No strategy data available")

            fig = go.Figure()
            
            # Strategy returns
            fig.add_trace(go.Scatter(
                x=strategy_data.index,
                y=strategy_data['cumulative_returns'],
                mode='lines',
                name='Strategy Returns',
                line=dict(color='green')
            ))
            
            # Benchmark if available
            if 'benchmark_returns' in strategy_data.columns:
                fig.add_trace(go.Scatter(
                    x=strategy_data.index,
                    y=strategy_data['benchmark_returns'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', dash='dash')
                ))
            
            fig.update_layout(
                title='Strategy Performance',
                xaxis_title='Date',
                yaxis_title='Cumulative Returns',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering strategy performance: {e}")
            st.error("Error rendering strategy chart")
    
    def render_trade_analysis(self, trades_data: pd.DataFrame):
        """Render trade analysis."""
        try:
            if trades_data.empty:
                st.warning("No trades data available")

            # Trade P&L distribution
            fig = px.histogram(
                trades_data,
                x='pnl',
                title='Trade P&L Distribution',
                nbins=30,
                color_discrete_sequence=['green']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering trade analysis: {e}")
            st.error("Error rendering trade analysis")

# --- Convenience Functions ---
def get_trading_dashboard() -> TradingDashboard:
    """Get a trading dashboard instance."""
    return TradingDashboard()

def get_strategy_dashboard() -> StrategyDashboard:
    """Get a strategy dashboard instance."""
    return StrategyDashboard()

def get_available_plugins() -> List[str]:
    """Get list of available dashboard plugins."""
    return get_plugin_manager().get_available_plugins()

def get_available_layouts() -> List[str]:
    """Get list of available Streamlit layouts."""
    return get_layout_manager().get_available_layouts()

def render_plugin(plugin_name: str, data: Dict[str, Any]) -> bool:
    """Render a specific plugin.
    
    Args:
        plugin_name: Name of the plugin to render
        data: Data to pass to the plugin
        
    Returns:
        True if successful, False otherwise
    """
    return get_plugin_manager().render_plugin(plugin_name, data)

def render_layout(layout_name: str, **kwargs) -> bool:
    """Render a specific layout.
    
    Args:
        layout_name: Name of the layout to render
        **kwargs: Arguments to pass to the layout
        
    Returns:
        True if successful, False otherwise
    """
    return get_layout_manager().render_layout(layout_name, **kwargs)