"""
Agent Registry for SemanticX Framework.
Provides agent registration and management capabilities.
"""
import logging
from typing import Dict, Type, Optional, List
from pathlib import Path
import json

from .example_agent import ExampleAgent
from ..core.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing all available agents.
    """
    
    def __init__(self):
        self.agents: Dict[str, Type[BaseAgent]] = {}
        self.agent_metadata: Dict[str, Dict] = {}
        self._register_default_agents()
        logger.info("AgentRegistry initialized")
    
    def _register_default_agents(self):
        """Register the default agents that come with the framework."""
        self.register_agent("example", ExampleAgent, {
            "description": "Example agent demonstrating framework capabilities",
            "domain": "general",
            "capabilities": ["framework_info", "agent_system", "tool_integration", "workflow_examples"],
            "example_queries": [
                "What is SemanticX Framework?",
                "How do I create an agent?",
                "How does tool integration work?",
                "Show me workflow examples"
            ]
        })
    
    def register_agent(self, name: str, agent_class: Type[BaseAgent], metadata: Dict = None):
        """
        Register a new agent.
        
        Args:
            name: Unique name for the agent
            agent_class: The agent class to register
            metadata: Optional metadata about the agent
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class must inherit from BaseAgent")
        
        self.agents[name] = agent_class
        self.agent_metadata[name] = metadata or {}
        logger.info(f"Registered agent: {name}")
    
    def get_agent(self, name: str) -> Optional[Type[BaseAgent]]:
        """Get an agent class by name."""
        return self.agents.get(name)
    
    def create_agent_instance(self, name: str, session_id: str, **kwargs) -> Optional[BaseAgent]:
        """Create an instance of an agent."""
        agent_class = self.get_agent(name)
        if agent_class:
            return agent_class(session_id=session_id, **kwargs)
        return None
    
    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self.agents.keys())
    
    def get_agent_info(self, name: str) -> Optional[Dict]:
        """Get metadata for a specific agent."""
        return self.agent_metadata.get(name)
    
    def unregister_agent(self, name: str):
        """Unregister an agent."""
        if name in self.agents:
            del self.agents[name]
            del self.agent_metadata[name]
            logger.info(f"Unregistered agent: {name}")


# Global agent registry instance
agent_registry = AgentRegistry()


def register_agent(name: str, agent_class: Type[BaseAgent], metadata: Dict = None):
    """Convenience function to register an agent."""
    agent_registry.register_agent(name, agent_class, metadata)


def get_agent(name: str) -> Optional[Type[BaseAgent]]:
    """Convenience function to get an agent class."""
    return agent_registry.get_agent(name)


def create_agent(name: str, session_id: str, **kwargs) -> Optional[BaseAgent]:
    """Convenience function to create an agent instance."""
    return agent_registry.create_agent_instance(name, session_id, **kwargs)


def list_agents() -> List[str]:
    """Convenience function to list all agents."""
    return agent_registry.list_agents()


def create_agent_boilerplate(agent_name: str, domain: str = None, description: str = None):
    """
    Create a new agent boilerplate with proper structure.
    
    Args:
        agent_name: Name for the new agent
        domain: Domain the agent operates in
        description: Description of the agent's purpose
    """
    # Create agent directory
    agent_dir = Path(f"agents/{agent_name.lower()}")
    agent_dir.mkdir(exist_ok=True)
    
    # Create agent file
    agent_file = agent_dir / f"{agent_name.lower()}_agent.py"
    
    # Generate agent code
    agent_code = f'''"""
{agent_name} Agent for SemanticX Framework.
{description or f"Specialized agent for {domain or 'general'} operations."}
"""
from typing import Dict, Any, List
import logging

from ...core.base_agent import BaseAgent
from ...models.state import ConversationState

logger = logging.getLogger(__name__)


class {agent_name}Agent(BaseAgent):
    """
    {agent_name} agent that handles {domain or 'general'} operations.
    
    {description or f"Specialized in {domain or 'general'} tasks and workflows."}
    """
    
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id, domain="{domain or 'general'}")
        logger.info(f"🎯 {agent_name}Agent initialized for session: {{session_id}}")

    def load_prompt(self) -> str:
        """Load the agent's system prompt."""
        return """You are a {domain or 'general'} agent for the SemanticX Framework.

Your role is to:
1. Handle {domain or 'general'} related requests
2. Process user queries efficiently
3. Use available tools when appropriate
4. Provide helpful and accurate responses

Be professional, helpful, and focused on {domain or 'general'} operations."""

    def load_tools(self) -> List[Dict[str, Any]]:
        """Load the agent's available tools."""
        # Get tools mapped to this agent
        return self.get_tools_for_agent("{agent_name}Agent")

    def get_valid_sub_intents(self) -> List[str]:
        """Return valid sub-intents for this agent."""
        return [
            "general_query",
            "specific_operation",
            "help_request"
        ]

    async def process(self, state: ConversationState) -> ConversationState:
        """
        Process the user's request.
        
        This is the main entry point where the agent handles user input
        and determines how to respond.
        """
        logger.info(f"{agent_name}Agent processing request for session: {{self.session_id}}")
        
        # Use base agent validation - only reset if needed
        state, was_reset = self.validate_and_reset_if_needed(state)
        if was_reset:
            logger.info(f"{agent_name}Agent state was reset due to invalid sub-intent")

        # Determine the user's intent if not already set (LLM-first with keyword fallback)
        if not state.sub_intent:
            state = await self.determine_intent_and_sub_intent(state, self._determine_sub_intent)

        # Process based on the determined sub-intent
        if state.sub_intent == "general_query":
            state = await self._handle_general_query(state)
        elif state.sub_intent == "specific_operation":
            state = await self._handle_specific_operation(state)
        else:
            # Default to help request
            state = await self._handle_help_request(state)

        return state

    async def _determine_sub_intent(self, state: ConversationState) -> ConversationState:
        """Determine the user's sub-intent based on their message."""
        if not state.conversation_history:
            state.sub_intent = "help_request"
            return state
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(state.conversation_history):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "").lower()
                break
        
        if not last_user_message:
            state.sub_intent = "help_request"
            return state
        
        # Simple keyword-based intent detection
        if any(word in last_user_message for word in ["help", "what", "how", "?"]):
            state.sub_intent = "help_request"
        elif any(word in last_user_message for word in ["do", "perform", "execute", "run"]):
            state.sub_intent = "specific_operation"
        else:
            state.sub_intent = "general_query"
        
        return state

    async def _handle_general_query(self, state: ConversationState) -> ConversationState:
        """Handle general queries."""
        response = f"I'm here to help with {domain or 'general'} operations. What would you like me to do?"
        state.add_assistant_message(response)
        return state

    async def _handle_specific_operation(self, state: ConversationState) -> ConversationState:
        """Handle specific operations."""
        # Use the base agent's LLM call with tools
        return await self._call_llm_with_tools(state)

    async def _handle_help_request(self, state: ConversationState) -> ConversationState:
        """Handle help requests."""
        response = f"I'm a {domain or 'general'} agent. I can help you with various operations. What do you need?"
        state.add_assistant_message(response)
        return state
'''
    
    # Write agent file
    with open(agent_file, 'w') as f:
        f.write(agent_code)
    
    # Create prompt file
    prompt_file = agent_dir / "prompt.txt"
    prompt_content = f"""You are a {domain or 'general'} agent for the SemanticX Framework.

Your role is to:
1. Handle {domain or 'general'} related requests
2. Process user queries efficiently
3. Use available tools when appropriate
4. Provide helpful and accurate responses

Be professional, helpful, and focused on {domain or 'general'} operations.

When users ask questions or request actions:
- Understand their intent clearly
- Use appropriate tools if available
- Provide clear, actionable responses
- Handle errors gracefully
- Ask for clarification when needed

Remember: You are specialized in {domain or 'general'} operations and should focus on helping users with related tasks."""
    
    with open(prompt_file, 'w') as f:
        f.write(prompt_content)
    
    # Create __init__.py for the agent package
    init_file = agent_dir / "__init__.py"
    init_content = f'''"""
{agent_name} Agent Package for SemanticX Framework.
"""
from .{agent_name.lower()}_agent import {agent_name}Agent

__all__ = ["{agent_name}Agent"]
'''
    
    with open(init_file, 'w') as f:
        f.write(init_content)
    
    # Create README for the agent
    readme_file = agent_dir / "README.md"
    readme_content = f'''# {agent_name} Agent

{description or f"Specialized agent for {domain or 'general'} operations."}

## Overview

This agent is designed to handle {domain or 'general'} related tasks and workflows within the SemanticX Framework.

## Capabilities

- General queries about {domain or 'general'} operations
- Specific {domain or 'general'} task execution
- Help and guidance for {domain or 'general'} workflows

## Usage

```python
from semanticx.agents.{agent_name.lower()} import {agent_name}Agent

# Create agent instance
agent = {agent_name}Agent(session_id="user123")

# Process user request
state = await agent.process(user_state)
```

## Configuration

The agent automatically loads tools mapped to "{agent_name}Agent" in the tool registry.

## Customization

Override the following methods to customize behavior:
- `load_prompt()` - Customize the agent's personality and capabilities
- `load_tools()` - Specify which tools the agent can use
- `process()` - Implement custom processing logic
- `get_valid_sub_intents()` - Define valid sub-intents for this agent
'''
    
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created {agent_name} agent boilerplate in {agent_dir}")
    print(f"📁 Files created:")
    print(f"   - {agent_file}")
    print(f"   - {prompt_file}")
    print(f"   - {init_file}")
    print(f"   - {readme_file}")
    print(f"\n🚀 Next steps:")
    print(f"   1. Customize the agent code in {agent_file}")
    print(f"   2. Modify the prompt in {prompt_file}")
    print(f"   3. Register the agent in your main application")
    print(f"   4. Test with: python -c \"from agents.{agent_name.lower()} import {agent_name}Agent; print('Agent created successfully!')\"")


# CLI command for creating new agents
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m agents create <agent_name> [domain] [description]")
        print("Example: python -m agents create CustomerService customer_service 'Handles customer inquiries'")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create":
        if len(sys.argv) < 3:
            print("Error: Agent name is required")
            sys.exit(1)
        
        agent_name = sys.argv[2]
        domain = sys.argv[3] if len(sys.argv) > 3 else None
        description = sys.argv[4] if len(sys.argv) > 4 else None
        
        create_agent_boilerplate(agent_name, domain, description)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: create")
        sys.exit(1)
