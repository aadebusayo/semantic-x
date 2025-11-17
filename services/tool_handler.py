"""
Tool Handler Service for SemanticX Framework.
Executes tools and validates arguments.
"""
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
import asyncio

logger = logging.getLogger(__name__)


class ToolHandler:
    """
    Handles tool execution and validation.
    """
    
    def __init__(self):
        self.registered_tools: Dict[str, callable] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        # Lazy import to avoid circular deps
        try:
            from ..core.tool_registry import ToolRegistry
            self._registry = ToolRegistry()
        except Exception:
            self._registry = None
        logger.info("ToolHandler initialized")
        from ..core.schema_validator import SchemaValidator
        self._schema_validator = SchemaValidator()
    
    def register_tool(self, name: str, func: callable, metadata: Dict[str, Any] = None):
        """Register a tool function."""
        self.registered_tools[name] = func
        self.tool_metadata[name] = metadata or {}
        logger.info(f"Registered tool: {name}")
    
    async def execute_tool(self, function_name: str, arguments: Dict[str, Any], user_state: Any = None) -> Dict[str, Any]:
        """
        Execute a tool with the given arguments.
        
        Args:
            function_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            user_state: User state context (optional)
            
        Returns:
            Dict containing execution result
        """
        try:
            logger.info(f"Executing tool: {function_name} with args: {arguments}")
            
            if function_name not in self.registered_tools:
                # Try executing schema-based HTTP tool
                if self._registry:
                    http_meta = self._registry.get_http_meta(function_name)
                    if http_meta:
                        from .http_tool_executor import HttpToolExecutor
                        executor = HttpToolExecutor()
                        http_result = await executor.execute(http_meta, arguments)
                        return {"success": True, "result": http_result, "tool_name": function_name}
                return {
                    "success": False,
                    "error": f"Tool '{function_name}' not found",
                    "error_type": "tool_not_found",
                    "retryable": False
                }
            
            # Validate arguments
            validated_args, error_msg = self.validate_arguments(function_name, arguments)
            if error_msg:
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": "validation_error",
                    "retryable": True
                }
            
            # Execute the tool
            tool_func = self.registered_tools[function_name]
            
            # Check if it's an async function
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**validated_args, user_state=user_state)
            else:
                result = tool_func(**validated_args, user_state=user_state)
            
            return {
                "success": True,
                "result": result,
                "tool_name": function_name,
                "execution_time": "completed"
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed for {function_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": "execution_error",
                "retryable": True
            }
    
    def validate_arguments(self, function_name: str, arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Validate tool arguments.
        
        Args:
            function_name: Name of the tool
            arguments: Arguments to validate
            
        Returns:
            Tuple of (validated_args, error_message)
        """
        if function_name not in self.registered_tools and not (self._registry and self._registry.get_tool_parameter_schema(function_name)):
            return {}, f"Tool '{function_name}' not found"
        
        arguments = arguments or {}
        schema = self._registry.get_tool_parameter_schema(function_name) if self._registry else None
        if schema:
            validated_args, error = self._schema_validator.validate(arguments, schema)
            if error:
                return {}, error
            return validated_args, None
        
        # Fallback basic validation
        validated_args = {k: v for k, v in arguments.items() if v is not None}
        return validated_args, None
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered tool."""
        if tool_name in self.registered_tools:
            return {
                "name": tool_name,
                "function": str(self.registered_tools[tool_name]),
                "metadata": self.tool_metadata.get(tool_name, {})
            }
        return None
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        names = list(self.registered_tools.keys())
        if self._registry:
            names.extend([n for n in self._registry.get_tool_names() if n not in names])
        return names
    
    def unregister_tool(self, tool_name: str):
        """Unregister a tool."""
        if tool_name in self.registered_tools:
            del self.registered_tools[tool_name]
            del self.tool_metadata[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")

    # Registry bridge helpers
    def get_all_tools(self) -> List[Dict[str, Any]]:
        if self._registry:
            return self._registry.get_all_tools()
        return []

    def get_tools_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        if self._registry:
            return self._registry.get_tools_for_agent(agent_name)
        return []

    def load_all_tools(self) -> List[Dict[str, Any]]:
        return self.get_all_tools()
