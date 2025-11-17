"""
Agent factory for SemanticX Framework.

Provides a centralized registry for agent classes so routing can dynamically
instantiate the correct agent for each plan step.
"""
from typing import Dict, Type
import logging

logger = logging.getLogger(__name__)


class AgentFactory:
    """Dynamic registry for agent classes."""

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, agent_cls: Type):
        """Register an agent class under a given name."""
        cls._registry[name] = agent_cls
        logger.info("Registered agent '%s' with class %s", name, agent_cls.__name__)

    @classmethod
    def create(cls, name: str, session_id: str):
        """Instantiate the requested agent."""
        agent_cls = cls._registry.get(name)
        if not agent_cls:
            raise ValueError(f"Agent '{name}' is not registered")
        return agent_cls(session_id=session_id)

    @classmethod
    def list_agents(cls):
        return list(cls._registry.keys())


# Register built-in agents
try:
    from agents.example_agent import ExampleAgent

    AgentFactory.register("ExampleAgent", ExampleAgent)
except Exception as exc:
    logger.warning("Failed to register ExampleAgent: %s", exc)

