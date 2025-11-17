"""
Interaction routing and complexity classification for SemanticX Framework.

Determines whether a request should be treated as simple vs. complex,
selects the appropriate LLM tier, and flags whether planning is required.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.state import ConversationState
from core.tool_registry import ToolRegistry
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Represents the routing outcome for a user request."""

    agent_name: str
    complexity: str
    requires_plan: bool
    model_tier: str
    model_name: Optional[str]
    target_tools: List[str]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "complexity": self.complexity,
            "requires_plan": self.requires_plan,
            "model_tier": self.model_tier,
            "model_name": self.model_name,
            "target_tools": self.target_tools,
            "reason": self.reason,
        }


class InteractionRouter:
    """
    Determines how to handle each user turn by combining heuristic rules,
    tool metadata, and intent signals stored in the conversation state.
    """

    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.tool_defaults_path = Path(getattr(settings, "tool_defaults_file", "./tool_defaults.json"))
        self.routing_config = self._load_tool_defaults()

    def _load_tool_defaults(self) -> Dict[str, Any]:
        """Load routing defaults from JSON configuration."""
        if not self.tool_defaults_path.exists():
            logger.warning("Tool defaults file %s not found. Using in-memory defaults.", self.tool_defaults_path)
            return {
                "model_tiers": {},
                "intents": {},
                "tools": {},
                "defaults": {
                    "complexity": "simple",
                    "requires_plan": False,
                    "preferred_model_tier": "simple",
                },
            }

        try:
            with open(self.tool_defaults_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                logger.info(
                    "Loaded routing defaults: %s intents, %s tool overrides",
                    len(data.get("intents", {})),
                    len(data.get("tools", {})),
                )
                return data
        except Exception as exc:
            logger.error("Failed to load tool defaults: %s", exc)
            return {
                "model_tiers": {},
                "intents": {},
                "tools": {},
                "defaults": {
                    "complexity": "simple",
                    "requires_plan": False,
                    "preferred_model_tier": "simple",
                },
            }

    def route(self, state: ConversationState, fallback_agent: Optional[str] = None) -> RoutingDecision:
        """
        Generate a routing decision for the provided state.
        """
        agent_name = self._determine_agent(state, fallback_agent)
        target_tools = self._resolve_candidate_tools(state, agent_name)
        intent_override = self._get_intent_metadata(state)
        routing_override = (state.metadata or {}).get("routing_overrides", {})

        complexity = intent_override.get("complexity") or self._infer_complexity_from_tools(target_tools)
        requires_plan = bool(intent_override.get("requires_plan", complexity == "complex"))
        preferred_tier = intent_override.get("preferred_model_tier") or self._get_preferred_tier_from_tools(target_tools, complexity)
        override_tier = routing_override.get("model_tier")
        if override_tier:
            preferred_tier = override_tier
        preferred_tier = self._enforce_security_policy(state, preferred_tier, target_tools)
        model_name = self._resolve_model_name(preferred_tier)
    def _enforce_security_policy(self, state: ConversationState, tier: str, tools: List[str]) -> str:
        """Force model tier overrides when handling sensitive data."""
        sensitive_domains = getattr(settings, "sensitive_data_domains", [])
        sensitive_tools = set(getattr(settings, "sensitive_tools", []))

        if state.domain in sensitive_domains:
            return "complex"

        if any(tool in sensitive_tools for tool in tools):
            return "complex"

        return tier

        reason = intent_override.get("reason") or self._build_reason(state, target_tools, complexity, requires_plan)
        if override_tier:
            reason += f"; override_model_tier={override_tier}"

        decision = RoutingDecision(
            agent_name=agent_name,
            complexity=complexity,
            requires_plan=requires_plan,
            model_tier=preferred_tier,
            model_name=model_name,
            target_tools=target_tools,
            reason=reason,
        )

        # Persist routing metadata on the state for downstream consumers.
        state.routing_metadata = decision.to_dict()
        state.available_tools = target_tools

        logger.info(
            "Routing decision for session %s: agent=%s complexity=%s model=%s requires_plan=%s reason=%s",
            state.session_id,
            agent_name,
            complexity,
            model_name or preferred_tier,
            requires_plan,
            reason,
        )
        try:
            from core.telemetry import record_event
            record_event(state, "routing_decision", {
                "agent": agent_name,
                "complexity": complexity,
                "requires_plan": requires_plan,
                "model": model_name or preferred_tier
            })
        except Exception:
            pass
        return decision

    def _determine_agent(self, state: ConversationState, fallback_agent: Optional[str]) -> str:
        """Determine which agent should handle this turn."""
        # Priority 1: Current plan step
        plan_step = state.get_current_plan_step()
        if plan_step and plan_step.get("agent"):
            agent_name = plan_step.get("agent")
            state.current_agent = agent_name
            return agent_name

        # Priority 2: Explicit metadata
        metadata_agent = (state.metadata or {}).get("target_agent")
        if metadata_agent:
            state.current_agent = metadata_agent
            return metadata_agent

        # Priority 3: Provided fallback
        if fallback_agent:
            state.current_agent = fallback_agent
            return fallback_agent

        # Priority 4: Config default
        default_agent = self.routing_config.get("defaults", {}).get("agent") or "ExampleAgent"
        state.current_agent = default_agent
        return default_agent

    def _get_intent_metadata(self, state: ConversationState) -> Dict[str, Any]:
        """Return metadata configured for the active intent/sub-intent."""
        intents_config = self.routing_config.get("intents", {})
        candidate_keys = [state.sub_intent, state.intent]
        for key in candidate_keys:
            if key and key in intents_config:
                meta = intents_config[key].copy()
                meta["reason"] = f"Matched configured intent '{key}'"
                return meta
        return {}

    def _resolve_candidate_tools(self, state: ConversationState, agent_name: str) -> List[str]:
        """
        Resolve which tools are most likely to be used for this turn.
        Priority order:
        1. Tools explicitly referenced in existing plan steps.
        2. Tools hinted via state metadata (requested_tool).
        3. Agent default tool mapping from ToolRegistry.
        """
        candidates: List[str] = []

        plan_step = state.get_current_plan_step()
        if plan_step:
            for tool in plan_step.get("tools", []):
                if isinstance(tool, str):
                    candidates.append(tool)
        elif state.plan:
            for step in state.plan or []:
                for tool in step.get("tools", []):
                    if isinstance(tool, str):
                        candidates.append(tool)

        requested_tool = state.metadata.get("requested_tool") if state.metadata else None
        if requested_tool:
            candidates.append(requested_tool)

        if not candidates:
            tools_for_agent = self.tool_registry.get_tools_for_agent(agent_name)
            for tool in tools_for_agent:
                function_def = tool.get("function", {})
                tool_name = function_def.get("name")
                if tool_name:
                    candidates.append(tool_name)

        if not candidates:
            # Fallback to every known tool so the orchestrator still has context.
            candidates = self.tool_registry.get_tool_names()

        # Deduplicate while preserving order
        seen = set()
        filtered = []
        for name in candidates:
            if name and name not in seen:
                seen.add(name)
                filtered.append(name)
        return filtered

    def _infer_complexity_from_tools(self, tools: List[str]) -> str:
        """Infer complexity classification by inspecting tool metadata."""
        tool_meta_map = self.routing_config.get("tools", {})
        for tool_name in tools:
            meta = tool_meta_map.get(tool_name, {})
            if meta.get("complexity") == "complex" or meta.get("requires_plan"):
                return "complex"
        return self.routing_config.get("defaults", {}).get("complexity", "simple")

    def _get_preferred_tier_from_tools(self, tools: List[str], default_complexity: str) -> str:
        """Derive preferred model tier from tool metadata."""
        tool_meta_map = self.routing_config.get("tools", {})
        for tool_name in tools:
            meta = tool_meta_map.get(tool_name, {})
            tier = meta.get("preferred_model_tier")
            if tier:
                return tier
        if default_complexity == "complex":
            return "complex"
        return self.routing_config.get("defaults", {}).get("preferred_model_tier", "simple")

    def _resolve_model_name(self, tier: str) -> Optional[str]:
        """Map a tier label to the concrete model name."""
        tier_config = self.routing_config.get("model_tiers", {}).get(tier)
        if tier_config:
            return tier_config.get("model")
        return None

    def _build_reason(self, state: ConversationState, tools: List[str], complexity: str, requires_plan: bool) -> str:
        """Generate a human-readable explanation for logs and debugging."""
        parts = []
        if state.sub_intent:
            parts.append(f"sub_intent={state.sub_intent}")
        if tools:
            parts.append(f"tools={','.join(tools[:3])}")
        parts.append(f"complexity={complexity}")
        parts.append(f"requires_plan={requires_plan}")
        return "; ".join(parts)

