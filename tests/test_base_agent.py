import pytest
import json

from core.base_agent import BaseAgent
from models.state import ConversationState


class SimpleAgent(BaseAgent):
    """Simple agent for testing BaseAgent functionality."""
    
    def load_prompt(self) -> str:
        return "You are a test agent."

    def load_tools(self):
        return []

    async def process(self, state: ConversationState) -> ConversationState:
        return await self._call_llm_only(state)


@pytest.mark.asyncio
async def test_base_agent_call_llm_only_adds_assistant_message():
    """Test that BaseAgent._call_llm_only() properly adds assistant messages to state."""
    print("\n" + "="*70)
    print("TEST: BaseAgent._call_llm_only() message handling")
    print("="*70)
    
    agent = SimpleAgent(session_id="test-session")
    print(f"✓ SimpleAgent created")
    print(f"  Session ID: {agent.session_id}")
    print(f"  Domain: {agent.domain}")
    
    state = ConversationState(session_id="test-session")
    print(f"\n✓ ConversationState initialized")
    print(f"  Session ID: {state.session_id}")
    print(f"  Conversation ID: {state.conversation_id}")
    print(f"  Initial message count: {len(state.conversation_history)}")
    
    # Add a user message so the agent has context
    state.add_user_message("Hello, can you help me?")
    print(f"\n→ Added user message")
    print(f"  Message count after user input: {len(state.conversation_history)}")
    print(f"  Last message: {state.conversation_history[-1].get('content', 'N/A')}")
    
    # Call the internal LLM-only path (no tools)
    print(f"\n→ Calling agent._call_llm_only(state)...")
    new_state = await agent._call_llm_only(state)
    
    print(f"\n✓ LLM call completed")
    print(f"  Final message count: {len(new_state.conversation_history)}")
    
    # After calling LLM-only, there should be at least one assistant message
    assistant_messages = new_state.get_assistant_messages()
    print(f"\n✓ Retrieved assistant messages")
    print(f"  Assistant message count: {len(assistant_messages)}")
    
    if assistant_messages:
        print(f"  Last assistant message: {assistant_messages[-1][:100]}...")
    
    assert len(assistant_messages) >= 1, f"Expected at least 1 assistant message, got {len(assistant_messages)}"
    print("✓ At least 1 assistant message present")
    
    assert isinstance(assistant_messages[-1], str), f"Expected str, got {type(assistant_messages[-1])}"
    print("✓ Assistant message is a string")
    
    # Verify conversation history structure
    print(f"\n✓ Conversation history structure:")
    for i, msg in enumerate(new_state.conversation_history):
        role = msg.get("role", "unknown")
        content_preview = msg.get("content", "N/A")[:50] if msg.get("content") else "(empty)"
        print(f"  [{i}] {role}: {content_preview}...")
    
    print("\n✅ TEST PASSED: BaseAgent._call_llm_only() properly adds messages")
    print("="*70 + "\n")
