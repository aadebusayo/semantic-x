# Agent Development Guide - SemanticX Framework

This guide explains how to create and customize agents in the SemanticX Framework using the new LangChain-style structure.

## 🏗️ New Agent Structure

The framework now follows a more organized, LangChain-inspired structure where each agent has its own directory with all related files:

```
agents/
├── example/                    # Example agent package
│   ├── __init__.py            # Package initialization
│   ├── example_agent.py       # Agent implementation
│   ├── prompt.txt             # Agent-specific prompt
│   └── README.md              # Agent documentation
├── customer_service/           # Customer service agent package
│   ├── __init__.py
│   ├── customer_service_agent.py
│   ├── prompt.txt
│   └── README.md
└── __init__.py                # Main agent registry
```

## 🚀 Creating New Agents

### Method 1: Using the CLI Command

The easiest way to create a new agent is using the built-in CLI command:

```bash
# Create a basic agent
python -m agents create CustomerService

# Create an agent with domain specification
python -m agents create CustomerService customer_service

# Create an agent with domain and description
python -m agents create CustomerService customer_service "Handles customer inquiries and support requests"
```

This will automatically create:
- Agent implementation file
- Agent-specific prompt file
- Package initialization file
- Comprehensive README

### Method 2: Manual Creation

You can also create agents manually by following this structure:

1. **Create agent directory**: `agents/your_agent_name/`
2. **Create agent file**: `your_agent_name_agent.py`
3. **Create prompt file**: `prompt.txt`
4. **Create package init**: `__init__.py`
5. **Register the agent**: Add to the agent registry

## 📝 Agent Implementation

### Basic Agent Template

```python
from typing import Dict, Any, List
import logging

from ...core.base_agent import BaseAgent
from ...models.state import ConversationState

logger = logging.getLogger(__name__)


class YourAgent(BaseAgent):
    """Your agent description."""
    
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id, domain="your_domain")
    
    def load_prompt(self) -> str:
        """Load the agent's system prompt."""
        return """Your agent prompt here..."""
    
    def load_tools(self) -> List[Dict[str, Any]]:
        """Load the agent's available tools."""
        return self.get_tools_for_agent("YourAgent")
    
    def get_valid_sub_intents(self) -> List[str]:
        """Return valid sub-intents for this agent."""
        return ["intent1", "intent2", "intent3"]
    
    async def process(self, state: ConversationState) -> ConversationState:
        """Process the user's request."""
        # Your agent logic here
        return await self._call_llm_with_tools(state)
```

### Required Methods

1. **`load_prompt()`** - Define your agent's personality and capabilities
2. **`load_tools()`** - Specify which tools your agent can use
3. **`process()`** - Handle user requests and determine responses

### Optional Methods

- **`get_valid_sub_intents()`** - Define valid sub-intents for this agent
- **Custom processing methods** - Add your own logic for specific intents

## 🔧 Tool Integration

### Automatic Tool Loading

Tools are automatically loaded from OpenAPI schemas in the `schemas/` directory:

```
schemas/
├── user_management.json       # Automatically becomes user_management tool
├── payment_processing.json    # Automatically becomes payment_processing tool
└── data_analytics.json       # Automatically becomes data_analytics tool
```

### Agent-Tool Mapping

Tools are mapped to agents using the `tool_mapping.json` file:

```json
{
  "YourAgent": [
    "user_management",
    "payment_processing"
  ],
  "default": [
    "user_management",
    "payment_processing",
    "data_analytics"
  ]
}
```

### Using Tools in Agents

```python
async def process(self, state: ConversationState) -> ConversationState:
    # Tools are automatically available based on mapping
    # Just call the base method and tools will be used if needed
    return await self._call_llm_with_tools(state)
```

## 📝 Prompt Management

### Agent-Specific Prompts

Each agent has its own prompt file (`prompt.txt`) that defines its personality and capabilities:

```
You are a customer service agent for the SemanticX Framework.

Your role is to:
1. Handle customer inquiries efficiently
2. Process support requests
3. Use available tools when appropriate
4. Provide helpful and accurate responses

Be professional, helpful, and focused on customer satisfaction.
```

### Dynamic Context Injection

Prompts support dynamic context injection using placeholders:

- **`{{llm_state}}`** - Current conversation state
- **`{{tool_schema}}`** - Available tools (automatically injected)

## 🎯 Best Practices

### 1. Keep Agents Focused

Each agent should have a specific domain or responsibility:

```python
# Good: Focused on customer service
class CustomerServiceAgent(BaseAgent):
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id, domain="customer_service")

# Avoid: Too broad
class GeneralAgent(BaseAgent):
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id, domain="everything")  # Too broad
```

### 2. Use Descriptive Sub-Intents

Define clear, descriptive sub-intents for your agent:

```python
def get_valid_sub_intents(self) -> List[str]:
    return [
        "account_inquiry",      # Handle account questions
        "payment_issue",        # Handle payment problems
        "technical_support",    # Handle technical issues
        "general_help"          # Handle general questions
    ]
```

### 3. Leverage Base Agent Features

Use the built-in functionality from BaseAgent:

```python
async def process(self, state: ConversationState) -> ConversationState:
    # Automatic validation and reset
    state, was_reset = self.validate_and_reset_if_needed(state)
    
    # Automatic intent detection
    if not state.sub_intent:
        state = await self._determine_sub_intent(state)
    
    # Use tools automatically
    return await self._call_llm_with_tools(state)
```

### 4. Handle Errors Gracefully

The framework provides built-in error handling:

```python
async def process(self, state: ConversationState) -> ConversationState:
    try:
        # Your logic here
        return await self._call_llm_with_tools(state)
    except Exception as e:
        # Error handling is automatic
        logger.error(f"Error in {self.__class__.__name__}: {e}")
        return state
```

## 🧪 Testing Your Agent

### 1. Test Import

```bash
python -c "from agents.your_agent import YourAgent; print('Agent created successfully!')"
```

### 2. Test Instantiation

```python
from agents.your_agent import YourAgent

agent = YourAgent(session_id="test123")
print(f"Agent created: {agent.__class__.__name__}")
print(f"Domain: {agent.domain}")
print(f"Available tools: {len(agent.load_tools())}")
```

### 3. Test Processing

```python
from models.state import ConversationState

# Create test state
state = ConversationState(
    session_id="test123",
    conversation_id="conv123",
    user_message="Hello, I need help"
)

# Process with agent
result = await agent.process(state)
print(f"Response: {result.get_last_assistant_message()}")
```

## 🔄 Agent Lifecycle

### 1. Registration

Agents are automatically registered when imported:

```python
# In your agent's __init__.py
from .your_agent import YourAgent

# The agent is automatically available
```

### 2. Instantiation

```python
from agents import create_agent

agent = create_agent("YourAgent", session_id="user123")
```

### 3. Processing

```python
# Process user requests
state = await agent.process(user_state)
```

### 4. Cleanup

Sessions are automatically managed by the framework.

## 📚 Example: Customer Service Agent

Here's a complete example of a customer service agent:

```python
# agents/customer_service/customer_service_agent.py
from typing import Dict, Any, List
import logging

from ...core.base_agent import BaseAgent
from ...models.state import ConversationState

logger = logging.getLogger(__name__)


class CustomerServiceAgent(BaseAgent):
    """Customer service agent for handling inquiries and support requests."""
    
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id, domain="customer_service")
    
    def load_prompt(self) -> str:
        return """You are a customer service agent for the SemanticX Framework.

Your role is to:
1. Handle customer inquiries efficiently
2. Process support requests
3. Use available tools when appropriate
4. Provide helpful and accurate responses

Be professional, helpful, and focused on customer satisfaction.
Always ask for clarification if a request is unclear."""
    
    def load_tools(self) -> List[Dict[str, Any]]:
        return self.get_tools_for_agent("CustomerServiceAgent")
    
    def get_valid_sub_intents(self) -> List[str]:
        return [
            "account_inquiry",
            "payment_issue", 
            "technical_support",
            "general_help"
        ]
    
    async def process(self, state: ConversationState) -> ConversationState:
        logger.info(f"CustomerServiceAgent processing request for session: {self.session_id}")
        
        # Use base agent validation
        state, was_reset = self.validate_and_reset_if_needed(state)
        
        # Determine intent if not set
        if not state.sub_intent:
            state = await self._determine_sub_intent(state)
        
        # Process based on intent
        if state.sub_intent == "account_inquiry":
            state = await self._handle_account_inquiry(state)
        elif state.sub_intent == "payment_issue":
            state = await self._handle_payment_issue(state)
        elif state.sub_intent == "technical_support":
            state = await self._handle_technical_support(state)
        else:
            state = await self._handle_general_help(state)
        
        return state
    
    async def _determine_sub_intent(self, state: ConversationState) -> ConversationState:
        """Determine the user's sub-intent."""
        if not state.conversation_history:
            state.sub_intent = "general_help"
            return state
        
        last_message = state.get_user_messages()[-1].lower()
        
        if any(word in last_message for word in ["account", "profile", "user"]):
            state.sub_intent = "account_inquiry"
        elif any(word in last_message for word in ["payment", "billing", "charge"]):
            state.sub_intent = "payment_issue"
        elif any(word in last_message for word in ["technical", "error", "bug", "problem"]):
            state.sub_intent = "technical_support"
        else:
            state.sub_intent = "general_help"
        
        return state
    
    async def _handle_account_inquiry(self, state: ConversationState) -> ConversationState:
        """Handle account-related inquiries."""
        return await self._call_llm_with_tools(state)
    
    async def _handle_payment_issue(self, state: ConversationState) -> ConversationState:
        """Handle payment-related issues."""
        return await self._call_llm_with_tools(state)
    
    async def _handle_technical_support(self, state: ConversationState) -> ConversationState:
        """Handle technical support requests."""
        return await self._call_llm_with_tools(state)
    
    async def _handle_general_help(self, state: ConversationState) -> ConversationState:
        """Handle general help requests."""
        response = "I'm here to help with customer service. What do you need assistance with?"
        state.add_assistant_message(response)
        return state
```

## 🚀 Next Steps

1. **Create your first agent** using the CLI command
2. **Customize the prompt** to match your use case
3. **Add your OpenAPI schemas** to the `schemas/` directory
4. **Test the agent** with sample conversations
5. **Deploy and iterate** based on user feedback

## 📖 Additional Resources

- [Framework Overview](../README.md)
- [API Reference](../docs/API.md)
- [Configuration Guide](../docs/CONFIGURATION.md)
- [Examples](../examples/)

---

**Happy Agent Building! 🎉**

The SemanticX Framework makes it easy to create powerful, intelligent agents with minimal code. Start building today!
