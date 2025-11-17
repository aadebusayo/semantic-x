"""
Universal BaseAgent class for SemanticX Framework.
Provides a foundation for building domain-specific agents with minimal code.
"""
from typing import Dict, Any, List, Optional
import json
import logging
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime, timezone

from models.state import ConversationState
from services.llm_service import LLMService
from services.tool_handler import ToolHandler
from utils.error_handler import IntelligentErrorHandler
from utils.prompt_utils import PromptManager
from core.orchestrator import Orchestrator
from core.retry_manager import RetryManager
from core.telemetry import record_event

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    
    This class provides a universal interface and integrates with the conversation state,
    LLM service, and tool system. Agents can be created with minimal code by extending
    this class and implementing the required abstract methods.
    """
    
    def __init__(self, session_id: str, domain: str = None):
        self.session_id = session_id
        self.domain = domain or self.__class__.__name__.replace('Agent', '').lower()
        
        # Initialize services
        self.llm_service = LLMService()
        self.tool_handler = ToolHandler()
        self.error_handler = IntelligentErrorHandler()
        self.prompt_manager = PromptManager()
        self.retry_manager = RetryManager()
        
        logger.info(f"🎯 {self.__class__.__name__} initialized for session: {session_id}")

    @abstractmethod
    def load_prompt(self) -> str:
        """
        Loads the agent's system prompt.
        
        Returns:
            str: The system prompt for this agent
        """
        raise NotImplementedError

    @abstractmethod
    def load_tools(self) -> List[Dict[str, Any]]:
        """
        Loads the agent's available tools in OpenAI format.
        
        Returns:
            List[Dict[str, Any]]: List of tools available to this agent
        """
        raise NotImplementedError

    @abstractmethod
    async def process(self, state: ConversationState) -> ConversationState:
        """
        The main entry point for an agent to process a user request.
        
        Args:
            state: The current conversation state
            
        Returns:
            ConversationState: The updated state after processing
        """
        raise NotImplementedError

    def get_valid_sub_intents(self) -> List[str]:
        """
        Returns a list of valid sub-intents for this agent.
        
        Should be overridden by each agent to define their specific sub-intents.
        
        Returns:
            List[str]: List of valid sub-intents
        """
        return []

    async def determine_intent_and_sub_intent(self, state: ConversationState, keyword_detector) -> ConversationState:
        """
        Determine intent/sub-intent using LLM first, with keyword fallback.
        keyword_detector is a callable(state) -> ConversationState that sets sub_intent.
        """
        try:
            last_user = state.get_user_messages()[-1] if state.get_user_messages() else ""
            if last_user:
                prompt = (
                    "You are an intent classifier. Given the user's latest message and short context, "
                    "identify primary intent and a fine-grained sub_intent from this agent's supported sub_intents. "
                    "Respond strictly as compact JSON: {\"intent\": string, \"sub_intent\": string}.\n\n"
                    f"Agent domain: {self.domain}\n"
                    f"Supported sub_intents: {self.get_valid_sub_intents()}\n"
                    f"User: {last_user}"
                )
                resp = await self.llm_service.generate_completion(messages=[{"role": "user", "content": prompt}])
                content = resp.get("content") or ""
                # Best-effort JSON parse
                data = None
                try:
                    data = json.loads(content)
                except Exception:
                    pass
                if isinstance(data, dict):
                    intent = data.get("intent")
                    sub_intent = data.get("sub_intent")
                    if isinstance(intent, str) and intent:
                        state.intent = intent
                    if isinstance(sub_intent, str) and sub_intent:
                        state.sub_intent = sub_intent
                        return state
        except Exception as e:
            logger.warning(f"LLM intent detection failed, falling back to keywords: {e}")

        # Fallback to keyword detector
        return await keyword_detector(state)

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Returns information about this agent.
        
        Returns:
            Dict[str, Any]: Agent information including name, description, capabilities
        """
        return {
            "name": self.__class__.__name__,
            "domain": self.domain,
            "description": self.__doc__ or f"{self.__class__.__name__} agent",
            "capabilities": self.get_valid_sub_intents(),
            "available_tools": [tool.get('function', {}).get('name') for tool in self.load_tools()]
        }

    def validate_and_reset_if_needed(self, state: ConversationState) -> tuple[ConversationState, bool]:
        """
        Validates if the current state is compatible with this agent.
        Only resets if there's an actual incompatibility.
        
        Args:
            state: The current conversation state
            
        Returns:
            tuple[ConversationState, bool]: (state, was_reset) tuple
        """
        valid_sub_intents = self.get_valid_sub_intents()
        needs_reset = False
        
        # Check if sub-intent is invalid for this agent
        if state.sub_intent and valid_sub_intents and state.sub_intent not in valid_sub_intents:
            logger.info(f"Sub-intent '{state.sub_intent}' not valid for {self.__class__.__name__}. Reset needed.")
            needs_reset = True
        
        if needs_reset:
            return self.reset_agent_state(state), True
        
        return state, False

    def reset_agent_state(self, state: ConversationState) -> ConversationState:
        """
        General reset mechanism for agent switching.
        Clears agent-specific state while preserving conversation history.
        
        Args:
            state: The current conversation state
            
        Returns:
            ConversationState: The reset state
        """
        logger.info(f"Resetting agent state for {self.__class__.__name__}")
        
        # Reset sub-intent if it's not valid for this agent
        valid_sub_intents = self.get_valid_sub_intents()
        if state.sub_intent and valid_sub_intents and state.sub_intent not in valid_sub_intents:
            logger.info(f"Sub-intent '{state.sub_intent}' not valid for {self.__class__.__name__}. Resetting.")
            state.sub_intent = None
        
        # Reset agent-specific flags and state
        state.current_step = None
        state.confirmation_pending = False
        state.auth_pending = False
        state.auth_method = None
        state.auth_context = {}
        state.is_task_complete = False
        
        # Clear any agent-specific metadata
        if hasattr(state, 'agent_metadata'):
            state.agent_metadata = {}
        
        return state
    def _set_model_override(self, state: ConversationState, tier: str):
        """Persist a preferred model tier override onto the state."""
        if not state.metadata:
            state.metadata = {}
        overrides = state.metadata.setdefault("routing_overrides", {})
        overrides["model_tier"] = tier

    def _get_model_override(self, state: ConversationState) -> Optional[str]:
        """Return the model override chosen by the routing layer, if any."""
        routing = getattr(state, "routing_metadata", None) or {}
        return routing.get("model_name")

    def _requires_plan(self, state: ConversationState) -> bool:
        """Return True if the routing layer flagged this request as requiring a plan."""
        routing = getattr(state, "routing_metadata", None) or {}
        return bool(routing.get("requires_plan"))

    async def _ensure_plan_if_needed(self, state: ConversationState) -> ConversationState:
        """
        Ensure a multi-step plan exists when the router flagged this request as complex.
        """
        self._invalidate_plan_if_context_changed(state)

        if not self._requires_plan(state):
            return state

        if state.plan:
            return state

        orchestrator = Orchestrator(session_id=self.session_id, domain=self.domain)
        plan = await orchestrator.create_plan(state)
        if plan:
            state.plan = plan
            if not state.current_step and isinstance(plan, list) and plan:
                first_step = plan[0]
                state.current_step = str(first_step.get("step"))
                state.current_agent = first_step.get("agent")
                state.current_task = first_step.get("action")
                state.auth_pending = bool(first_step.get("requires_auth"))
                state.auth_method = first_step.get("auth_method")
            state.plan_context = {
                "intent": state.intent,
                "sub_intent": state.sub_intent,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            record_event(state, "plan_created", {"step_count": len(plan), "agent": state.current_agent})
            logger.info("Generated plan with %s steps for session %s", len(plan), self.session_id)
        else:
            logger.warning("Router requested a plan but orchestrator returned none for session %s", self.session_id)
        return state

    def _invalidate_plan_if_context_changed(self, state: ConversationState):
        """Invalidate existing plan if intent/sub-intent changed since plan creation."""
        if not state.plan:
            return
        plan_ctx = state.plan_context or {}
        if plan_ctx.get("intent") != state.intent or plan_ctx.get("sub_intent") != state.sub_intent:
            logger.info(
                "Invalidating plan for session %s due to context change (old intent=%s/%s, new intent=%s/%s)",
                self.session_id,
                plan_ctx.get("intent"),
                plan_ctx.get("sub_intent"),
                state.intent,
                state.sub_intent,
            )
            state.clear_plan(reason="intent_changed")

    def _prepare_messages(self, state: ConversationState, prompt_override: str = None) -> List[Dict[str, Any]]:
        """
        Prepares the message history for the LLM call.
        
        Args:
            state: The current conversation state
            prompt_override: Optional prompt to use instead of the agent's default
            
        Returns:
            List[Dict[str, Any]]: Formatted messages for LLM
        """
        # Use the override if provided, otherwise fall back to the agent's default prompt
        prompt = prompt_override or self.load_prompt()
        messages = [{"role": "system", "content": prompt}]
        
        # Use the serializable version of the history
        messages.extend(state.get_serializable_history())
        return messages

    async def _call_llm_with_tools(self, state: ConversationState, prompt_override: str = None) -> ConversationState:
        """
        Calls the LLM with the current conversation history and available tools.
        
        Handles tool calls and responses, updating the state.
        
        Args:
            state: The current conversation state
            prompt_override: Optional prompt to use instead of the agent's default
            
        Returns:
            ConversationState: The updated state
        """
        tools = self.load_tools()
        if not tools:
            logger.warning(f"No tools available for {self.__class__.__name__}")
            # Call LLM without tools
            return await self._call_llm_only(state, prompt_override)

        # Check if we're in service_unavailable mode and handle gracefully
        if state.auth_status == "service_unavailable":
            logger.info(f"Session {self.session_id} is in service_unavailable mode - providing limited functionality")
            state.add_assistant_message(
                "I understand you'd like to access your information, but I'm currently experiencing technical difficulties with our services. I can still help you with general questions, or you can try again later when the service is restored."
            )
            return state

        messages = self._prepare_messages(state, prompt_override=prompt_override)

        model_override = self._get_model_override(state)

        try:
            llm_response = await self.llm_service.generate_completion(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                model_override=model_override,
            )
            record_event(state, "llm_call", {"model": model_override or self.llm_service.model, "tools_used": bool(tools)})

            tool_calls = llm_response.get("tool_calls")
            if tool_calls:
                state.last_function_call = llm_response
                return await self._handle_tool_calls(state, tool_calls)
            else:
                # No tool call, just a regular response
                content = llm_response.get("content", "I'm sorry, I didn't get a response.")
                state.add_assistant_message(content)
                return state

        except Exception as e:
            logger.error(f"BaseAgent Error: LLM call failed for agent {self.__class__.__name__}: {e}", exc_info=True)
            error_message = await self.error_handler.get_user_friendly_error_message(e, "llm_call")
            state.add_assistant_message(error_message)
            state.set_error(str(e))
            return state

    async def _call_llm_only(self, state: ConversationState, prompt_override: str = None) -> ConversationState:
        """
        Calls the LLM without tools for simple responses.
        
        Args:
            state: The current conversation state
            prompt_override: Optional prompt to use instead of the agent's default
            
        Returns:
            ConversationState: The updated state
        """
        messages = self._prepare_messages(state, prompt_override=prompt_override)

        try:
            llm_response = await self.llm_service.generate_completion(
                messages=messages,
                model_override=self._get_model_override(state),
            )
            record_event(state, "llm_call", {"model": self._get_model_override(state) or self.llm_service.model, "tools_used": False})
            content = llm_response.get("content", "I'm sorry, I didn't get a response.")
            state.add_assistant_message(content)
            return state

        except Exception as e:
            logger.error(f"BaseAgent Error: LLM call failed for agent {self.__class__.__name__}: {e}", exc_info=True)
            error_message = await self.error_handler.get_user_friendly_error_message(e, "llm_call")
            state.add_assistant_message(error_message)
            state.set_error(str(e))
            return state

    async def _handle_tool_calls(self, state: ConversationState, tool_calls: List[Dict]) -> ConversationState:
        """
        Executes tool calls requested by the LLM and updates the state.
        
        Args:
            state: The current conversation state
            tool_calls: List of tool calls to execute
            
        Returns:
            ConversationState: The updated state
        """
        # Add the assistant's decision to call tools to the conversation history
        serializable_tool_calls = [self._serialize_tool_call(tc) for tc in tool_calls]
        assistant_message = {"role": "assistant", "tool_calls": serializable_tool_calls}

        if state.last_function_call and state.last_function_call.get("content"):
            assistant_message["content"] = state.last_function_call.get("content")
        
        state.conversation_history.append(assistant_message)
        
        tool_outputs = []
        for tool_call in tool_calls:
            try:
                function_name = tool_call.function.name
                tool_call_id = tool_call.id
                arguments = json.loads(tool_call.function.arguments)
                
                logger.info(f'Executing tool: {function_name} with args: {arguments}')
                
                # Execute the tool
                result = await self.tool_handler.execute_tool(
                    function_name=function_name,
                    arguments=arguments,
                    user_state=state
                )
                
                # Handle the result
                if result.get("success", True):
                    tool_result_content = json.dumps(result)
                    state.is_task_complete = True  # Signal that the step is complete
                    self.retry_manager.reset(state)
                    record_event(state, "tool_success", {"tool": function_name})
                else:
                    retryable = result.get("retryable", True)
                    record_event(state, "tool_failure", {"tool": function_name, "retryable": retryable})
                    tool_result_content = json.dumps({
                        "error": True,
                        "user_message": result.get("message", "An error occurred"),
                        "error_type": result.get("error_type", "tool_execution_error"),
                        "retryable": retryable
                    })
                    decision = self.retry_manager.handle_failure(state, function_name, retryable)
                    if decision.message == "auth_refresh_required":
                        state.auth_pending = True
                        state.auth_context["reason"] = "validation_failed"
                        state.confirmation_pending = False
                        record_event(state, "auth_retry_required", {"tool": function_name})
                        logger.info("Auth refresh requested for session %s after tool failure", self.session_id)
                        return state
                    if decision.escalate_model and decision.new_model_tier:
                        self._set_model_override(state, decision.new_model_tier)
                        record_event(state, "model_escalated", {"new_tier": decision.new_model_tier})
                        logger.info("Escalating model tier to %s due to repeated failures", decision.new_model_tier)
                    if decision.message:
                        state.metadata.setdefault("system_events", []).append(decision.message)

            except json.JSONDecodeError as e:
                logger.error(f"BaseAgent Error: Failed to decode JSON for tool {function_name} arguments: {e}", exc_info=True)
                tool_result_content = json.dumps({
                    "error": True, 
                    "user_message": "There was a technical issue with the tool's parameters. Please try again.",
                    "error_type": "internal_error", 
                    "retryable": True
                })
            except Exception as e:
                logger.error(f"BaseAgent Error: Unhandled exception in tool call for {function_name}: {e}", exc_info=True)
                error_content = await self.error_handler.get_user_friendly_error_message(e, "tool_execution")
                tool_result_content = json.dumps({
                    "error": True, 
                    "user_message": error_content, 
                    "error_type": "internal_error", 
                    "retryable": True
                })

            # Always append a result for each tool call, even if it's an error
            tool_outputs.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": function_name,
                "content": tool_result_content,
            })
        
        state.conversation_history.extend(tool_outputs)

        # After executing tools, call the LLM again to get a natural language response
        try:
            messages_for_final_response = self._prepare_messages(state)
            final_response = await self.llm_service.generate_completion(
                messages=messages_for_final_response,
                model_override=self._get_model_override(state),
            )
            record_event(state, "llm_call", {"model": self._get_model_override(state) or self.llm_service.model, "tools_used": False})
            state.add_assistant_message(final_response.get("content", "Tool execution completed."))
            state.is_task_complete = True  # Signal that the step is complete
        except Exception as e:
            logger.error(f"BaseAgent Error: LLM call failed after tool execution for agent {self.__class__.__name__}: {e}", exc_info=True)
            error_message = await self.error_handler.get_user_friendly_error_message(e, "llm_call_after_tool")
            state.add_assistant_message(error_message)

        return state

    def _serialize_tool_call(self, tool_call) -> Dict[str, Any]:
        """Safely serialize a tool call object."""
        if hasattr(tool_call, 'dict'):
            return tool_call.dict()
        elif isinstance(tool_call, dict):
            return tool_call
        else:
            return {"function": {"name": str(tool_call), "arguments": "{}"}}

    def get_tools_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Get tools available for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List[Dict[str, Any]]: List of available tools
        """
        return self.tool_handler.get_tools_for_agent(agent_name)

    def validate_tool_arguments(self, function_name: str, arguments: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[str]]:
        """
        Validate tool arguments before execution.
        
        Args:
            function_name: Name of the function to validate
            arguments: Arguments to validate
            
        Returns:
            tuple[Dict[str, Any], Optional[str]]: (validated_args, error_message)
        """
        return self.tool_handler.validate_arguments(function_name, arguments)

    async def summarize_conversation(self, state: ConversationState) -> str:
        """
        Generate a summary of the current conversation.
        
        Args:
            state: The current conversation state
            
        Returns:
            str: Conversation summary
        """
        if not state.conversation_history:
            return "No conversation to summarize."
        
        try:
            summary_prompt = f"""
            Please provide a concise summary of the following conversation. 
            Focus on the main intent, key actions taken, and final outcome.
            
            Conversation:
            {json.dumps(state.conversation_history, indent=2)}
            
            Summary:
            """
            
            response = await self.llm_service.generate_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200,
                model_override=self._get_model_override(state),
            )
            
            return response.get("content", "Unable to generate summary.")
            
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")
            return "Error generating conversation summary."

    def get_agent_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this agent for logging and monitoring.
        
        Returns:
            Dict[str, Any]: Agent metadata
        """
        return {
            "agent_class": self.__class__.__name__,
            "session_id": self.session_id,
            "domain": self.domain,
            "valid_sub_intents": self.get_valid_sub_intents(),
            "available_tools_count": len(self.load_tools())
        }
