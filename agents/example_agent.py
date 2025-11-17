"""
Example Agent for SemanticX Framework.
Demonstrates how to create a simple agent with minimal code.
"""
from typing import Dict, Any, List
import logging

from ..core.base_agent import BaseAgent
from ..models.state import ConversationState

logger = logging.getLogger(__name__)


class ExampleAgent(BaseAgent):
    """
    Example agent that demonstrates the framework's capabilities.
    
    This agent shows how easy it is to create new agents by extending BaseAgent
    and implementing just a few required methods.
    """
    
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id, domain="example")
        logger.info(f"🎯 ExampleAgent initialized for session: {session_id}")

    def load_prompt(self) -> str:
        """Load the agent's system prompt."""
        return """You are a helpful example agent for the SemanticX Framework.

Your role is to:
1. Help users understand how the framework works
2. Demonstrate the agent system capabilities
3. Provide clear, helpful responses
4. Show how tools can be integrated

Be friendly, informative, and always ready to help users learn about the framework.

When users ask about specific topics, use these guidelines:

**Framework Info**: Explain the core concepts, architecture, and benefits
**Agent System**: Describe how agents work, how to create them, and their capabilities
**Tool Integration**: Show how tools are automatically loaded and used
**Workflow Examples**: Provide concrete examples of how the framework handles workflows
**General Help**: Offer guidance on getting started and best practices"""

    def load_tools(self) -> List[Dict[str, Any]]:
        """Load the agent's available tools."""
        # This agent doesn't need any specific tools, but you could add them here
        return []

    def get_valid_sub_intents(self) -> List[str]:
        """Return valid sub-intents for this agent."""
        return [
            "framework_info",
            "agent_system",
            "tool_integration", 
            "workflow_examples",
            "general_help"
        ]

    async def process(self, state: ConversationState) -> ConversationState:
        """
        Process the user's request.
        
        This is the main entry point where the agent handles user input
        and determines how to respond.
        """
        logger.info(f"ExampleAgent processing request for session: {self.session_id}")
        
        # Use base agent validation - only reset if needed
        state, was_reset = self.validate_and_reset_if_needed(state)
        if was_reset:
            logger.info("ExampleAgent state was reset due to invalid sub-intent")

        # Determine the user's intent if not already set
        if not state.sub_intent:
            # LLM-first, fallback to keyword
            state = await self.determine_intent_and_sub_intent(state, self._determine_sub_intent)

        # Ensure routing requirements like planning are satisfied before handling sub-intent
        state = await self._ensure_plan_if_needed(state)

        # Process based on the determined sub-intent
        if state.sub_intent == "framework_info":
            state = await self._handle_framework_info(state)
        elif state.sub_intent == "agent_system":
            state = await self._handle_agent_system(state)
        elif state.sub_intent == "tool_integration":
            state = await self._handle_tool_integration(state)
        elif state.sub_intent == "workflow_examples":
            state = await self._handle_workflow_examples(state)
        else:
            # Default to general help
            state = await self._handle_general_help(state)

        return state

    async def _determine_sub_intent(self, state: ConversationState) -> ConversationState:
        """Determine the user's sub-intent based on their message."""
        if not state.conversation_history:
            state.sub_intent = "general_help"
            return state
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(state.conversation_history):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "").lower()
                break
        
        if not last_user_message:
            state.sub_intent = "general_help"
            return state
        
        # Simple keyword-based intent detection
        if any(word in last_user_message for word in ["framework", "semanticx", "what is"]):
            state.sub_intent = "framework_info"
        elif any(word in last_user_message for word in ["agent", "how to create", "agent system"]):
            state.sub_intent = "agent_system"
        elif any(word in last_user_message for word in ["tool", "api", "integration"]):
            state.sub_intent = "tool_integration"
        elif any(word in last_user_message for word in ["workflow", "example", "show me"]):
            state.sub_intent = "workflow_examples"
        else:
            state.sub_intent = "general_help"
        
        return state

    async def _handle_framework_info(self, state: ConversationState) -> ConversationState:
        """Handle framework information requests."""
        response = """🎯 **SemanticX Framework Overview**

SemanticX is a universal, domain-agnostic AI agent orchestration framework that provides a solid foundation for building intelligent conversational AI systems.

**Key Features:**
• **Universal Agent Architecture** - Pluggable agent system with minimal code requirements
• **Intelligent Orchestration** - Multi-step planning and workflow management  
• **Dynamic Tool Integration** - Automatic API schema parsing into function tools
• **State Management** - Robust conversation state tracking with Pydantic validation
• **Error Handling** - LLM-driven intelligent error analysis and recovery

**Architecture:**
The framework consists of core components like BaseAgent, Orchestrator, ToolRegistry, and SessionManager that work together to provide a complete AI agent solution.

**Benefits:**
- Works with any domain (banking, healthcare, e-commerce, etc.)
- Create new agents with minimal code
- Automatic tool integration from OpenAPI schemas
- Production-ready with FastAPI and WebSocket support

Would you like me to explain any specific aspect in more detail?"""
        
        state.add_assistant_message(response)
        return state

    async def _handle_agent_system(self, state: ConversationState) -> ConversationState:
        """Handle agent system questions."""
        response = """🤖 **Agent System in SemanticX**

The agent system is designed to be simple yet powerful. Here's how it works:

**Creating Agents:**
```python
from semanticx.core.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def load_prompt(self) -> str:
        return "You are a helpful assistant specialized in..."
    
    def load_tools(self) -> List[Dict[str, Any]]:
        return self.get_tools_for_agent("MyCustomAgent")
    
    async def process(self, state: ConversationState) -> ConversationState:
        # Your agent logic here
        return await self._call_llm_with_tools(state)
```

**Key Components:**
• **BaseAgent** - Abstract base class with built-in functionality
• **load_prompt()** - Define your agent's personality and capabilities
• **load_tools()** - Specify which tools your agent can use
• **process()** - Handle user requests and determine responses

**Agent Lifecycle:**
1. User sends message
2. Agent determines intent and sub-intent
3. Agent processes request using tools if needed
4. Agent responds and updates conversation state

**Benefits:**
- Minimal code required (just 3 methods)
- Automatic tool integration
- Built-in error handling and state management
- Consistent interface across all agents

Would you like me to show you how to create a specific type of agent?"""
        
        state.add_assistant_message(response)
        return state

    async def _handle_tool_integration(self, state: ConversationState) -> ConversationState:
        """Handle tool integration questions."""
        response = """🔧 **Tool Integration in SemanticX**

Tool integration is completely automatic and requires no manual setup!

**How It Works:**
1. **Place OpenAPI Schemas** in the `schemas/` directory
2. **Framework Automatically** converts them to function tools
3. **Tools Are Mapped** to agents based on configuration
4. **Agents Can Use** tools without any additional code

**Example Schema Structure:**
```
schemas/
├── user_management.json
├── payment_processing.json
└── data_analytics.json
```

**Automatic Conversion:**
The framework reads OpenAPI schemas and converts them to OpenAI function format:
```json
{
  "type": "function",
  "function": {
    "name": "user_management",
    "description": "Manage user accounts",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {"type": "string", "description": "Action to perform"},
        "user_id": {"type": "string", "description": "User identifier"}
      }
    }
  }
}
```

**Agent-Tool Mapping:**
- Tools are automatically available to agents
- You can customize which tools each agent can access
- No manual registration or setup required

**Benefits:**
- Zero configuration needed
- Automatic parameter validation
- Built-in error handling
- Easy to add new capabilities

Would you like me to show you how to create and use custom tools?"""
        
        state.add_assistant_message(response)
        return state

    async def _handle_workflow_examples(self, state: ConversationState) -> ConversationState:
        """Handle workflow examples requests."""
        response = """📋 **Workflow Examples in SemanticX**

Here are some real-world examples of how SemanticX handles complex workflows:

**Example 1: Customer Support Workflow**
```
User: "I need help with my account"
1. Intent Detection → Customer Support Agent
2. Account Lookup → User Management Tool
3. Issue Classification → Support Ticket Tool
4. Resolution Planning → Orchestrator
5. Response Generation → Support Agent
```

**Example 2: E-commerce Order Processing**
```
User: "I want to place an order"
1. Intent Detection → Shopping Agent
2. Product Search → Catalog Tool
3. Inventory Check → Inventory Tool
4. Order Creation → Order Management Tool
5. Payment Processing → Payment Tool
6. Confirmation → Shopping Agent
```

**Example 3: Data Analysis Workflow**
```
User: "Analyze my sales data"
1. Intent Detection → Analytics Agent
2. Data Retrieval → Data Warehouse Tool
3. Analysis Planning → Orchestrator
4. Statistical Analysis → Analytics Tools
5. Report Generation → Reporting Tool
6. Insights Delivery → Analytics Agent
```

**Key Workflow Features:**
• **Multi-step Planning** - LLM-driven workflow creation
• **Dynamic Execution** - Adapts based on results
• **Error Recovery** - Intelligent handling of failures
• **Progress Tracking** - Real-time status updates
• **Tool Coordination** - Seamless tool chaining

**Benefits:**
- Complex workflows with simple agent definitions
- Automatic error handling and recovery
- Real-time progress tracking
- Easy to modify and extend

Would you like me to walk through creating a specific workflow?"""
        
        state.add_assistant_message(response)
        return state

    async def _handle_general_help(self, state: ConversationState) -> ConversationState:
        """Handle general help requests."""
        response = """🌟 **Welcome to SemanticX Framework!**

I'm here to help you understand and use the SemanticX Framework effectively.

**Getting Started:**
1. **Explore the Framework** - Ask me about core concepts and architecture
2. **Learn About Agents** - Understand how to create and customize agents
3. **Discover Tools** - See how automatic tool integration works
4. **See Examples** - Get real-world workflow examples

**What You Can Ask Me:**
• "What is SemanticX Framework?"
• "How do I create a new agent?"
• "How does tool integration work?"
• "Show me workflow examples"
• "What are the benefits of this framework?"

**Quick Facts:**
- **Domain Agnostic** - Works with any business domain
- **Minimal Code** - Create agents with just 3 methods
- **Automatic Tools** - No manual tool registration needed
- **Production Ready** - Built for real-world deployment
- **Easy Extension** - Simple to add new capabilities

**Next Steps:**
Start by asking me about any specific aspect of the framework that interests you, or let me guide you through the basics!

What would you like to learn about first?"""
        
        state.add_assistant_message(response)
        return state
