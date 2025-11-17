import pytest
import json

from services.tool_handler import ToolHandler


def sample_echo(input: str, user_state=None):
    """Simple echo tool for testing."""
    return {"echo": input}


@pytest.mark.asyncio
async def test_tool_handler_register_and_execute_sync_tool():
    """Test that ToolHandler can register and execute a synchronous tool."""
    print("\n" + "="*70)
    print("TEST: ToolHandler.register_tool() and execute_tool()")
    print("="*70)
    
    th = ToolHandler()
    print(f"✓ ToolHandler initialized")
    print(f"  Registered tools: {th.list_tools()}")
    
    # Register a test tool
    print(f"\n→ Registering 'echo' tool...")
    th.register_tool("echo", sample_echo, metadata={"description": "Echo tool"})
    print(f"✓ Tool registered")
    print(f"  Registered tools: {th.list_tools()}")
    
    # Get tool info
    tool_info = th.get_tool_info("echo")
    print(f"\n✓ Retrieved tool info:")
    print(f"  Name: {tool_info.get('name', 'N/A')}")
    print(f"  Metadata: {tool_info.get('metadata', {})}")
    
    # Execute the tool
    print(f"\n→ Executing 'echo' tool with input 'hello'...")
    resp = await th.execute_tool("echo", {"input": "hello"})
    
    print(f"\n✓ Tool execution result:")
    print(f"  Success: {resp.get('success', 'N/A')}")
    print(f"  Tool name: {resp.get('tool_name', 'N/A')}")
    print(f"  Result type: {type(resp.get('result', {})).__name__}")
    print(f"  Result content: {json.dumps(resp.get('result', {}))}")
    
    assert isinstance(resp, dict), f"Expected dict response, got {type(resp)}"
    print("✓ Response is a dict")
    
    assert resp.get("success") is True, f"Expected success=True, got {resp.get('success')}"
    print("✓ Tool execution was successful")
    
    assert isinstance(resp.get("result"), dict), f"Expected dict result, got {type(resp.get('result'))}"
    print("✓ Result is a dict")
    
    assert resp.get("result").get("echo") == "hello", f"Expected echo='hello', got {resp.get('result')}"
    print("✓ Echo tool produced correct output")
    
    # Cleanup: unregister the tool
    print(f"\n→ Unregistering 'echo' tool...")
    th.unregister_tool("echo")
    print(f"✓ Tool unregistered")
    
    # Verify it's gone
    tool_info_after = th.get_tool_info("echo")
    print(f"  Tool info after unregister: {tool_info_after}")
    
    assert tool_info_after is None, f"Expected tool to be unregistered, but got: {tool_info_after}"
    print("✓ Tool successfully removed from registry")
    
    print("\n✅ TEST PASSED: ToolHandler registration and execution")
    print("="*70 + "\n")
