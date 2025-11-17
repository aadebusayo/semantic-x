"""
Dynamic Tool Registry for SemanticX Framework.
Automatically loads OpenAPI schemas and maps them to agents.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import asyncio
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Dynamic tool registry that automatically loads OpenAPI schemas
    and maps them to agents based on configuration.
    """
    
    def __init__(self, schema_dir: str = "./schemas", tool_mapping_file: str = "./tool_mapping.json"):
        self.schema_dir = Path(schema_dir)
        self.tool_mapping_file = Path(tool_mapping_file)
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_http_meta: Dict[str, Dict[str, Any]] = {}
        self.tool_parameter_schemas: Dict[str, Dict[str, Any]] = {}
        self.agent_tool_mapping: Dict[str, List[str]] = {}
        self._load_tools()
        self._load_agent_mapping()
        self._start_refresh_task()
    
    def _load_tools(self):
        """Load all OpenAPI schemas and convert them to function tools."""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory {self.schema_dir} does not exist")
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema = json.load(f)
                
                tool_name = schema_file.stem
                tool_def, http_meta = self._convert_schema_to_tool(schema, tool_name)
                self.tools[tool_name] = tool_def
                if tool_def.get("type") == "function":
                    params = tool_def.get("function", {}).get("parameters")
                    if params:
                        self.tool_parameter_schemas[tool_name] = params
                if http_meta:
                    self.tool_http_meta[tool_name] = http_meta
                logger.info(f"Loaded tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")
        
        self._tool_version = datetime.utcnow().isoformat()
    
    def _convert_schema_to_tool(self, schema: Dict[str, Any], tool_name: str) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Convert OpenAPI schema to OpenAI function tool format."""
        # Extract the first path operation as the main function
        paths = schema.get('paths', {})
        if not paths:
            return self._create_default_tool(tool_name), None
        
        # Get the first available path and method
        path, path_item = next(iter(paths.items()))
        method = next(iter(path_item.keys()))
        operation = path_item[method]
        
        # Extract parameters and responses
        parameters = operation.get('parameters', [])
        request_body = operation.get('requestBody', {})
        # Save HTTP metadata
        servers = schema.get('servers', [])
        http_meta = {
            "method": method.upper(),
            "path": path,
            "servers": servers,
        }
        
        # Convert to OpenAI function format
        function_def = {
            "name": tool_name,
            "description": operation.get('summary', f"Execute {tool_name} operation"),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        
        # Add parameters
        for param in parameters:
            if param.get('in') == 'query':
                param_name = param['name']
                param_schema = param.get('schema', {})
                function_def["parameters"]["properties"][param_name] = {
                    "type": param_schema.get('type', 'string'),
                    "description": param.get('description', f"Parameter {param_name}")
                }
                if param.get('required', False):
                    function_def["parameters"]["required"].append(param_name)
        
        # Add request body if present
        if request_body and 'content' in request_body:
            content = request_body['content']
            if 'application/json' in content:
                body_schema = content['application/json'].get('schema', {})
                if body_schema.get('type') == 'object':
                    for prop_name, prop_schema in body_schema.get('properties', {}).items():
                        function_def["parameters"]["properties"][prop_name] = {
                            "type": prop_schema.get('type', 'string'),
                            "description": prop_schema.get('description', f"Body parameter {prop_name}")
                        }
                    if body_schema.get('required'):
                        function_def["parameters"]["required"].extend(body_schema['required'])
        
        return ({
            "type": "function",
            "function": function_def
        }, http_meta)
    
    def _create_default_tool(self, tool_name: str) -> Dict[str, Any]:
        """Create a default tool when schema parsing fails."""
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"Execute {tool_name} operation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": f"Input for {tool_name} operation"
                        }
                    },
                    "required": ["input"]
                }
            }
        }
    
    def _load_agent_mapping(self):
        """Load agent-tool mapping configuration."""
        if not self.tool_mapping_file.exists():
            logger.info(f"Tool mapping file {self.tool_mapping_file} not found, using default mapping")
            self._create_default_mapping()
            return
        
        try:
            with open(self.tool_mapping_file, 'r') as f:
                self.agent_tool_mapping = json.load(f)
            logger.info(f"Loaded tool mapping for {len(self.agent_tool_mapping)} agents")
        except Exception as e:
            logger.error(f"Failed to load tool mapping: {e}")
            self._create_default_mapping()
    
    def _create_default_mapping(self):
        """Create default agent-tool mapping."""
        # Map all tools to all agents by default
        all_tools = list(self.tools.keys())
        self.agent_tool_mapping = {
            "default": all_tools,
            "planner": all_tools,
            "orchestrator": all_tools
        }
        
        # Save the default mapping
        self._save_mapping()
    
    def _save_mapping(self):
        """Save the current agent-tool mapping."""
        try:
            with open(self.tool_mapping_file, 'w') as f:
                json.dump(self.agent_tool_mapping, f, indent=2)
            logger.info(f"Saved tool mapping to {self.tool_mapping_file}")
        except Exception as e:
            logger.error(f"Failed to save tool mapping: {e}")
    
    def get_tools_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get tools available for a specific agent."""
        # Try agent-specific mapping first
        if agent_name in self.agent_tool_mapping:
            tool_names = self.agent_tool_mapping[agent_name]
        else:
            # Fall back to default mapping
            tool_names = self.agent_tool_mapping.get("default", [])
        
        # Return actual tool objects
        return [self.tools.get(name, {}) for name in tool_names if name in self.tools]
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools."""
        return list(self.tools.values())
    
    def get_tool_names(self) -> List[str]:
        """Get names of all available tools."""
        return list(self.tools.keys())

    def get_http_meta(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get HTTP execution metadata for a tool if available."""
        return self.tool_http_meta.get(tool_name)
    
    def add_tool_mapping(self, agent_name: str, tool_names: List[str]):
        """Add or update tool mapping for an agent."""
        self.agent_tool_mapping[agent_name] = tool_names
        self._save_mapping()
        logger.info(f"Updated tool mapping for {agent_name}: {tool_names}")
    
    def reload_tools(self):
        """Reload tools from schema directory."""
        self.tools.clear()
        self.tool_http_meta.clear()
        self.tool_parameter_schemas.clear()
        self._load_tools()
        logger.info("Reloaded tools from schema directory")
    
    def get_tool_schema_summary(self) -> str:
        """Get a summary of available tools for prompt injection."""
        if not self.tools:
            return "No tools available"
        
        summary = f"Available tools (schema_version={self._tool_version}):\n"
        for tool_name, tool_def in self.tools.items():
            if tool_def.get('type') == 'function':
                function = tool_def.get('function', {})
                summary += f"- {tool_name}: {function.get('description', 'No description')}\n"
        
        return summary

    def get_tool_parameter_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Return the JSON schema for tool parameters if available."""
        return self.tool_parameter_schemas.get(tool_name)

    def _start_refresh_task(self):
        interval = getattr(settings, "tool_schema_refresh_interval_seconds", 0)
        if interval <= 0:
            return
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._refresh_periodically(interval))
        except RuntimeError:
            logger.warning("Event loop unavailable; tool refresh task not started")

    async def _refresh_periodically(self, interval: int):
        while True:
            try:
                await asyncio.sleep(interval)
                self.reload_tools()
                self._load_agent_mapping()
                logger.info("Periodic tool refresh completed")
            except Exception as exc:
                logger.error("Error during tool refresh: %s", exc)
