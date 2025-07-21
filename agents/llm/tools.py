"""Tool registry for LLM agents with dynamic tool loading and execution."""

import asyncio
import importlib
import inspect
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    required_parameters: List[str]
    return_type: str
    is_async: bool = False
    category: str = "general"
    version: str = "1.0.0"


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: str = "general",
    version: str = "1.0.0",
):
    """Decorator to register a function as a tool.

    Args:
        name: Optional name for the tool
        description: Optional description of the tool
        category: Tool category
        version: Tool version
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> ToolResult:
            start_time = asyncio.get_event_loop().time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return ToolResult(
                    success=True,
                    data=result,
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}")
                return ToolResult(
                    success=False,
                    data=None,
                    error=str(e),
                    execution_time=asyncio.get_event_loop().time() - start_time,
                )

        # Get function metadata
        sig = inspect.signature(func)
        parameters = {}
        required_parameters = []

        for name, param in sig.parameters.items():
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )
            parameters[name] = {
                "type": str(param_type),
                "default": (
                    param.default if param.default != inspect.Parameter.empty else None
                ),
            }
            if param.default == inspect.Parameter.empty:
                required_parameters.append(name)

        # Create tool metadata
        wrapper.metadata = ToolMetadata(
            name=name or func.__name__,
            description=description or func.__doc__ or "",
            parameters=parameters,
            required_parameters=required_parameters,
            return_type=str(sig.return_annotation),
            is_async=asyncio.iscoroutinefunction(func),
            category=category,
            version=version,
        )

        return wrapper

    return decorator


class ToolRegistry:
    """Registry for managing and executing tools."""

    def __init__(self, tools_dir: Optional[Union[str, Path]] = None):
        """Initialize the tool registry.

        Args:
            tools_dir: Optional directory containing tool definitions
        """
        self.tools: Dict[str, Callable] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.tools_dir = Path(tools_dir) if tools_dir else None

        if self.tools_dir:
            self._load_tools_from_dir()

        return {
            "success": True,
            "message": "Initialization completed",
            "timestamp": datetime.now().isoformat(),
        }

    def _load_tools_from_dir(self) -> None:
        """Load tools from the tools directory."""
        if not self.tools_dir.exists():
            logger.warning(f"Tools directory {self.tools_dir} does not exist")

        for file in self.tools_dir.glob("*.py"):
            try:
                # Import the module
                module_name = file.stem
                spec = importlib.util.spec_from_file_location(module_name, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find and register tools
                for name, obj in inspect.getmembers(module):
                    if hasattr(obj, "metadata"):
                        self.register_tool(obj)

            except Exception as e:
                logger.error(f"Failed to load tools from {file}: {str(e)}")

    def register_tool(self, tool_func: Callable) -> None:
        """Register a tool function.

        Args:
            tool_func: The tool function to register
        """
        if not hasattr(tool_func, "metadata"):
            raise ValueError("Tool function must be decorated with @tool")

        metadata = tool_func.metadata
        self.tools[metadata.name] = tool_func
        self.metadata[metadata.name] = metadata
        logger.info(f"Registered tool: {metadata.name}")

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool

        Returns:
            ToolResult containing the execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_func = self.tools[tool_name]
        metadata = self.metadata[tool_name]

        # Validate required parameters
        missing_params = [
            param for param in metadata.required_parameters if param not in args
        ]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")

        # Execute the tool
        try:
            result = await tool_func(**args)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"traceback": traceback.format_exc()},
            )

    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolMetadata if the tool exists, None otherwise
        """
        return self.metadata.get(tool_name)

    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered tools.

        Args:
            category: Optional category to filter by

        Returns:
            List of tool metadata dictionaries
        """
        tools = []
        for name, metadata in self.metadata.items():
            if category and metadata.category != category:
                continue
            tools.append(
                {
                    "name": name,
                    "description": metadata.description,
                    "parameters": metadata.parameters,
                    "required_parameters": metadata.required_parameters,
                    "return_type": metadata.return_type,
                    "is_async": metadata.is_async,
                    "category": metadata.category,
                    "version": metadata.version,
                }
            )
        return tools

    def export_tools(self, path: Union[str, Path]) -> None:
        """Export tool definitions to a file.

        Args:
            path: Path to save the tool definitions
        """
        tools_data = {
            name: {
                "description": metadata.description,
                "parameters": metadata.parameters,
                "required_parameters": metadata.required_parameters,
                "return_type": metadata.return_type,
                "is_async": metadata.is_async,
                "category": metadata.category,
                "version": metadata.version,
            }
            for name, metadata in self.metadata.items()
        }

        with open(path, "w") as f:
            json.dump(tools_data, f, indent=2)

    def import_tools(self, path: Union[str, Path]) -> None:
        """Import tool definitions from a file.

        Args:
            path: Path to load tool definitions from
        """
        with open(path, "r") as f:
            tools_data = json.load(f)

        for name, data in tools_data.items():
            metadata = ToolMetadata(
                name=name,
                description=data["description"],
                parameters=data["parameters"],
                required_parameters=data["required_parameters"],
                return_type=data["return_type"],
                is_async=data["is_async"],
                category=data["category"],
                version=data["version"],
            )
            self.metadata[name] = metadata
