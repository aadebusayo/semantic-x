# SemanticX Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Version: 0.1.0-alpha](https://img.shields.io/badge/Version-0.1.0--alpha-blue)](./ROADMAP.md)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://github.com/psf/black)

A universal, domain-agnostic AI agent orchestration framework that provides a solid foundation for building intelligent conversational AI systems.

## 🎯 Quick Links

📚 **Documentation & Resources**
- [AGENT_DEVELOPMENT.md](./AGENT_DEVELOPMENT.md) - How to create and customize agents
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Contributing guidelines and PR checklist
- [ROADMAP.md](./ROADMAP.md) - Feature roadmap and release plan
- [Interactive Demo Notebook](./notebooks/demo_agent.ipynb) - Run code examples in Jupyter

## 🚀 Core Features

- **Universal Agent Architecture** - Pluggable agent system with minimal code requirements
- **Intelligent Orchestration** - Multi-step planning and workflow management
- **Dynamic Tool Integration** - Automatic API schema parsing into function tools
- **State Management** - Robust conversation state tracking with Pydantic validation
- **Error Handling** - LLM-driven intelligent error analysis and recovery
- **Session Management** - Conversation lifecycle and memory management
- **Real-time Communication** - WebSocket-based real-time messaging
- **Prompt Engineering** - Dynamic context injection and template management

## 🏗️ Architecture Overview

```
SemanticX/
├── AGENT_DEVELOPMENT.md       # Agent development guide
├── agents/                    # Agent implementations and registry
│   ├── __init__.py            # Agent registry/CLI boilerplate
│   └── example_agent.py       # Example agent
├── api/
│   └── router.py              # REST + WebSocket routes
├── config.py                  # Configuration management
├── core/                      # Core framework components
│   ├── __init__.py
│   ├── base_agent.py          # Universal agent base class
│   ├── memory_manager.py      # Minimal memory manager hooks
│   ├── orchestrator.py        # Workflow orchestration engine
│   ├── session_manager.py     # Session lifecycle management
│   └── tool_registry.py       # Dynamic tool loading system
├── env.example                # Sample environment configuration
├── FRAMEWORK_SUMMARY.md       # Implementation summary
├── LICENSE                    # MIT license
├── main.py                    # Application entry point (FastAPI)
├── models/                    # Data models
│   ├── __init__.py
│   ├── schemas.py             # API message schemas
│   └── state.py               # Conversation state model
├── prompts/
│   ├── __init__.py
│   └── orchestrator/
│       └── planner.txt        # Planner prompt template
├── requirements.txt           # Dependencies
├── schemas/                   # OpenAPI tool schemas
│   └── Planner.json
├── services/                  # Core services
│   ├── __init__.py
│   ├── http_tool_executor.py  # Generic HTTP executor for tools
│   ├── llm_service.py         # LLM integration (stub)
│   ├── tool_handler.py        # Tool execution engine
│   └── vector_store.py        # Vector store (Qdrant + memory)
├── tool_mapping.json          # Agent-tool mapping
└── utils/                     # Utilities
    ├── __init__.py
    ├── error_handler.py       # Intelligent error handling
    └── prompt_utils.py        # Prompt loading/injection
```

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Your First Agent

The easiest way is using the built-in CLI command:

```bash
# Create a basic agent
python -m agents create CustomerService

# Create with domain and description
python -m agents create CustomerService customer_service "Handles customer inquiries"
```

Or manually create the agent structure:

```python
from semanticx.core.base_agent import BaseAgent
from semanticx.models.state import ConversationState

class MyCustomAgent(BaseAgent):
    def load_prompt(self) -> str:
        return "You are a helpful assistant specialized in..."
    
    def load_tools(self) -> List[Dict[str, Any]]:
        return self.get_tools_for_agent("MyCustomAgent")
    
    async def process(self, state: ConversationState) -> ConversationState:
        # Your agent logic here
        return await self._call_llm_with_tools(state)
```

### 3. Agent Structure

Each agent gets its own directory with all related files:

```
agents/my_custom/
├── my_custom_agent.py    # Agent implementation
├── prompt.txt            # Agent-specific prompt
├── __init__.py           # Package initialization
└── README.md             # Documentation
```

### 4. Run the Framework
```bash
python main.py
```

## 🔧 Configuration

The framework is highly configurable through environment variables and configuration files:

- `LLM_PROVIDER` - Choose your LLM provider (OpenAI, Azure, etc.)
- `VECTOR_STORE_TYPE` - Select vector database (memory, Qdrant)
- `TOOL_SCHEMA_DIR` - Directory containing your API schemas
- `AGENT_PROMPT_DIR` - Directory for agent-specific prompts
  - Optional Qdrant config: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`

## 🎨 Customization Points

### Agent System
- Extend `BaseAgent` for custom agent behavior
- Override `load_prompt()` and `load_tools()` methods
- Implement custom `process()` logic

### Tool Integration
- Place OpenAPI schemas in `schemas/` directory
- Tools are automatically loaded and mapped to agents
- Custom tool handlers can be registered

### Prompt System
- Each agent has its own prompt file (`prompt.txt`)
- Use `{{llm_state}}` and `{{tool_schema}}` for dynamic injection
- Prompts are organized by agent in their respective directories

### Memory System
- Extend memory classes for custom storage backends
- Implement conversation summarization and embedding
- Configure vector storage for semantic search

## 🌟 Key Benefits

1. **Domain Agnostic** - Works with any domain (banking, healthcare, e-commerce, etc.)
2. **Minimal Code** - Create new agents with just a few lines of code
3. **Automatic Tool Loading** - API schemas automatically become available tools
4. **Intelligent Error Handling** - LLM-driven error analysis and recovery
5. **Scalable Architecture** - Built for production-scale deployments
6. **Real-time Communication** - WebSocket-based real-time messaging
7. **Extensible Design** - Easy to add new capabilities and integrations

## 📚 Examples

See the `agents/` directory for complete agent implementations and the `AGENT_DEVELOPMENT.md` guide for detailed development instructions.

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines for more details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
