"""
Universal Orchestrator Engine for SemanticX Framework.
Manages multi-step planning and workflow orchestration for any domain.
"""
import logging
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime, timezone

from models.state import ConversationState
from services.llm_service import LLMService
from utils.prompt_utils import PromptManager
from core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Universal workflow orchestration engine.
    
    This class creates structured, multi-step plans for AI agents to follow
    based on user requests. It's domain-agnostic and can be configured
    for any use case.
    """
    
    def __init__(self, session_id: str, domain: str = None):
        self.session_id = session_id
        self.domain = domain or "general"
        self.llm_service = LLMService()
        self.prompt_manager = PromptManager()
        self.tool_registry = ToolRegistry()
        
        logger.info(f"🎯 Orchestrator initializing for session: {session_id} in domain: {self.domain}")
        
        # Load the planner prompt template
        self.planner_prompt_template = self._load_planner_prompt()
        
        # Load the Planner schema
        self.planner_schema = self._load_planner_schema()

    def _load_planner_prompt(self) -> str:
        """Load the planner prompt template."""
        try:
            # Try to load domain-specific planner prompt
            prompt = self.prompt_manager.load_prompt(f"orchestrator/planner_{self.domain}")
            logger.info(f"Loaded domain-specific planner prompt for {self.domain}")
        except FileNotFoundError:
            try:
                # Fall back to general planner prompt
                prompt = self.prompt_manager.load_prompt("orchestrator/planner")
                logger.info("Loaded general planner prompt")
            except FileNotFoundError:
                # Use fallback prompt
                prompt = self._get_fallback_planner_prompt()
                logger.warning("Using fallback planner prompt")
        
        return prompt

    def _get_fallback_planner_prompt(self) -> str:
        """Get a fallback planner prompt if none is found."""
        return f"""You are an intelligent workflow planner for {self.domain} operations.

Your task is to create a structured, multi-step plan for handling user requests.
Analyze the user's intent and break it down into logical, sequential steps.

IMPORTANT: You must respond with valid JSON in this exact format:
{{
    "plan": [
        {{
            "step": "step_number",
            "action": "description_of_action",
            "agent": "agent_name",
            "tools": ["tool1", "tool2"],
            "expected_outcome": "what_should_happen",
            "dependencies": ["previous_step_if_any"]
        }}
    ],
    "estimated_duration": "estimated_time",
    "complexity": "low|medium|high",
    "notes": "additional_considerations"
}}

Guidelines:
- Keep steps simple and focused
- Ensure logical flow and dependencies
- Specify which agent should handle each step
- List required tools for each step
- Consider error handling and fallbacks
- Estimate realistic duration and complexity

Current context: {{llm_state}}

Available tools: {{tool_schema}}

Create a plan that will efficiently accomplish the user's goal."""

    def _load_planner_schema(self) -> Dict[str, Any]:
        """Load the Planner schema for tool calling."""
        try:
            # Try to load domain-specific planner schema
            schema_path = f"schemas/Planner_{self.domain.capitalize()}.json"
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                logger.info(f"Loaded domain-specific planner schema: {schema_path}")
                return schema
        except Exception as e:
            logger.warning(f"Could not load domain-specific planner schema: {e}")
        
        try:
            # Fall back to general planner schema
            schema_path = "schemas/Planner.json"
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    schema = json.load(f)
                logger.info("Loaded general planner schema")
                return schema
        except Exception as e:
            logger.warning(f"Could not load general planner schema: {e}")
        
        # Return empty schema if none found
        return {}

    async def create_plan(self, state: ConversationState, replan_mode: bool = False) -> List[Dict[str, str]]:
        """
        Creates a structured, multi-step plan for the AI to follow based on the user's request.
        
        Args:
            state: The current conversation state
            replan_mode: Whether this is a replanning request
            
        Returns:
            List[Dict[str, str]]: List of plan steps
        """
        logger.info(f"Orchestrator creating a plan for session: {self.session_id}")

        # If we're in service_unavailable mode, don't create plans that require authentication
        if state.auth_status == "service_unavailable":
            logger.info(f"Session {self.session_id} is in service_unavailable mode - not creating authentication-dependent plans")
            return []

        # Create dynamic prompt with current state
        prompt = self._create_dynamic_prompt(state)
        
        if replan_mode:
            prompt += '\n\nIMPORTANT: This is a replanning request. The user has deviated from the previous plan. Adjust dynamically based on the latest input and current context.'
        
        try:
            messages = [{"role": "system", "content": prompt}]
            
            # Add recent conversation context
            recent_messages = state.get_recent_messages(5)  # Last 5 messages for context
            messages.extend(recent_messages)
            
            response = await self.llm_service.generate_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )

            plan_json = response.get("content", "{}")
            
            # Parse the plan
            plan_data = json.loads(plan_json)
            plan = plan_data.get("plan", [])

            if not plan:
                logger.warning("Planner returned an empty plan.")
                return []
            
            # Validate plan structure
            validated_plan = self._validate_plan(plan)
            
            logger.info(f"Orchestrator created plan with {len(validated_plan)} steps: {validated_plan}")
            return validated_plan

        except json.JSONDecodeError:
            logger.error(f"Orchestrator Error: Failed to decode JSON from planner response: {plan_json}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Orchestrator Error: An unexpected error occurred during plan creation: {e}", exc_info=True)
            return []

    def _create_dynamic_prompt(self, state: ConversationState) -> str:
        """Create a dynamic prompt with current state context."""
        prompt = self.planner_prompt_template
        
        # Inject the current state summary
        state_summary = self._create_state_summary(state)
        prompt = prompt.replace("{{llm_state}}", json.dumps(state_summary, indent=2))
        
        # Inject the available tools from the tool registry
        available_tools = self.tool_registry.get_tool_schema_summary()
        prompt = prompt.replace("{{tool_schema}}", available_tools)
        
        return prompt

    def _create_state_summary(self, state: ConversationState) -> Dict[str, Any]:
        """Create a summary of the current state for the planner."""
        summary = {
            "session_id": state.session_id,
            "conversation_id": state.conversation_id,
            "current_intent": state.intent,
            "sub_intent": state.sub_intent,
            "domain": state.domain,
            "auth_status": state.auth_status,
            "current_step": state.current_step,
            "is_task_complete": state.is_task_complete,
            "confirmation_pending": state.confirmation_pending,
            "auth_pending": state.auth_pending,
            "error_message": state.error_message,
            "available_tools": state.available_tools,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add user profile summary if available
        if state.user_profile:
            summary["user_profile"] = {
                "user_id": state.user_id,
                "has_profile": bool(state.user_profile),
                "profile_keys": list(state.user_profile.keys()) if isinstance(state.user_profile, dict) else []
            }
        
        # Add conversation context
        if state.conversation_history:
            summary["conversation_context"] = {
                "message_count": len(state.conversation_history),
                "recent_user_messages": state.get_user_messages()[-3:],  # Last 3 user messages
                "has_tool_calls": any("tool_calls" in msg for msg in state.conversation_history)
            }
        
        return summary

    def _validate_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Validate and clean the plan structure."""
        validated_plan = []
        
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                logger.warning(f"Invalid step {i}: not a dictionary")
                continue
            
            # Ensure required fields
            validated_step = {
                "step": str(step.get("step", i + 1)),
                "action": str(step.get("action", "Unknown action")),
                "agent": str(step.get("agent", "Unknown")),
                "tools": step.get("tools", []),
                "expected_outcome": str(step.get("expected_outcome", "Step completion")),
                "dependencies": step.get("dependencies", [])
            }
            
            # Validate tools list
            if not isinstance(validated_step["tools"], list):
                validated_step["tools"] = []
            
            # Validate dependencies list
            if not isinstance(validated_step["dependencies"], list):
                validated_step["dependencies"] = []
            
            validated_plan.append(validated_step)
        
        return validated_plan

    def reset_for_agent_switch(self, state: ConversationState, new_intent: str = None) -> ConversationState:
        """
        General reset mechanism for when switching between agents.
        This provides a clean slate while preserving conversation history.
        
        Args:
            state: The current conversation state
            new_intent: The new intent to switch to
            
        Returns:
            ConversationState: The reset state
        """
        logger.info(f"Orchestrator resetting state for agent switch. New intent: {new_intent}")
        
        # Reset intent-specific state
        if new_intent:
            state.intent = new_intent
        
        # Reset sub-intent and step tracking
        state.sub_intent = None
        state.current_step = None
        state.current_task = None
        
        # Reset confirmation and auth flags
        state.confirmation_pending = False
        state.auth_pending = False
        
        # Clear any pending plan steps that might be agent-specific
        if state.plan:
            state.plan = []
        
        # Clear any agent-specific metadata
        if hasattr(state, 'agent_metadata'):
            state.agent_metadata = {}
        
        logger.info(f"State reset complete for agent switch to intent: {new_intent}")
        return state

    async def evaluate_plan_progress(self, state: ConversationState) -> Dict[str, Any]:
        """
        Evaluate the progress of the current plan.
        
        Args:
            state: The current conversation state
            
        Returns:
            Dict[str, Any]: Progress evaluation
        """
        if not state.plan:
            return {
                "has_plan": False,
                "progress": 0.0,
                "current_step": None,
                "remaining_steps": 0,
                "estimated_completion": None
            }
        
        total_steps = len(state.plan)
        current_step_index = 0
        
        # Find current step
        if state.current_step:
            for i, step in enumerate(state.plan):
                if step.get("step") == state.current_step:
                    current_step_index = i
                    break
        
        progress = (current_step_index / total_steps) * 100 if total_steps > 0 else 0
        remaining_steps = total_steps - current_step_index
        
        return {
            "has_plan": True,
            "progress": round(progress, 1),
            "current_step": state.current_step,
            "remaining_steps": remaining_steps,
            "total_steps": total_steps,
            "estimated_completion": self._estimate_completion_time(remaining_steps)
        }

    def _estimate_completion_time(self, remaining_steps: int) -> str:
        """Estimate completion time based on remaining steps."""
        if remaining_steps == 0:
            return "Complete"
        elif remaining_steps <= 2:
            return "Almost complete"
        elif remaining_steps <= 5:
            return "In progress"
        else:
            return "Early stages"

    async def optimize_plan(self, state: ConversationState, feedback: str) -> List[Dict[str, str]]:
        """
        Optimize the current plan based on user feedback or performance data.
        
        Args:
            state: The current conversation state
            feedback: User feedback or performance data
            
        Returns:
            List[Dict[str, str]]: Optimized plan
        """
        logger.info(f"Orchestrator optimizing plan based on feedback: {feedback}")
        
        if not state.plan:
            logger.warning("No plan to optimize")
            return []
        
        try:
            optimization_prompt = f"""
            The current plan needs optimization based on this feedback: {feedback}
            
            Current plan:
            {json.dumps(state.plan, indent=2)}
            
            Please provide an optimized version of this plan that addresses the feedback.
            Return the plan in the same JSON format.
            """
            
            messages = [{"role": "system", "content": optimization_prompt}]
            
            response = await self.llm_service.generate_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            
            plan_json = response.get("content", "{}")
            plan_data = json.loads(plan_json)
            optimized_plan = plan_data.get("plan", [])
            
            if optimized_plan:
                logger.info("Plan optimization completed successfully")
                return self._validate_plan(optimized_plan)
            else:
                logger.warning("Plan optimization returned empty result")
                return state.plan
                
        except Exception as e:
            logger.error(f"Error optimizing plan: {e}")
            return state.plan

    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the orchestrator."""
        return {
            "session_id": self.session_id,
            "domain": self.domain,
            "has_planner_prompt": bool(self.planner_prompt_template),
            "has_planner_schema": bool(self.planner_schema),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
