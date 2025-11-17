import pytest
import json

from services.llm_service import LLMService


@pytest.mark.asyncio
async def test_generate_completion_returns_content(caplog):
    """LLMService should return a dict containing a content string even when provider is not configured."""
    print("\n" + "="*70)
    print("TEST: LLMService.generate_completion() basic behavior")
    print("="*70)
    
    svc = LLMService()
    print(f"✓ LLMService initialized")
    print(f"  Provider: {svc.provider}")
    print(f"  Model: {svc.model}")
    
    messages = [{"role": "user", "content": "hello"}]
    print(f"\n→ Calling generate_completion with: {json.dumps(messages)}")
    
    resp = await svc.generate_completion(messages=messages)
    
    print(f"\n✓ Response received:")
    print(f"  Type: {type(resp)}")
    print(f"  Keys: {list(resp.keys())}")
    print(f"  Content preview: {resp.get('content', 'N/A')[:100]}...")
    
    assert isinstance(resp, dict), f"Expected dict, got {type(resp)}"
    print("✓ Response is a dict")
    
    assert "content" in resp, f"Expected 'content' key, got keys: {list(resp.keys())}"
    print("✓ Response contains 'content' key")
    
    assert isinstance(resp["content"], str), f"Expected str content, got {type(resp['content'])}"
    print("✓ Content is a string")
    
    print("\n✅ TEST PASSED: LLMService.generate_completion()")
    print("="*70 + "\n")


@pytest.mark.asyncio
async def test_test_connection_returns_bool(caplog):
    """LLMService.test_connection should return a boolean."""
    print("\n" + "="*70)
    print("TEST: LLMService.test_connection() behavior")
    print("="*70)
    
    svc = LLMService()
    print(f"✓ LLMService initialized")
    print(f"  Provider: {svc.provider}")
    
    print(f"\n→ Calling test_connection()...")
    ok = await svc.test_connection()
    
    print(f"\n✓ Connection test result: {ok}")
    print(f"  Type: {type(ok)}")
    
    assert isinstance(ok, bool), f"Expected bool, got {type(ok)}"
    print("✓ Result is a boolean")
    
    print("\n✅ TEST PASSED: LLMService.test_connection()")
    print("="*70 + "\n")
