"""Unit test: LLM failure returns structured error response, not exception."""
import pytest
from unittest.mock import AsyncMock, patch
from litellm.exceptions import Timeout


@pytest.mark.asyncio
async def test_llm_timeout_returns_error_dict():
    """When LLM times out, agent should return a structured error dict, not raise."""
    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.side_effect = Timeout(message="Request timed out", model="test-model", llm_provider="test")

        from agent import SampleAgent
        agent = SampleAgent()
        result = await agent._run_classification(
            "Some feedback text.",
            context_id="test-005"
        )

    assert "error" in result
    assert result["sentiment"] is None
    assert result["confidence"] is None
    assert result["reasoning"] is None


@pytest.mark.asyncio
async def test_llm_non_json_response_returns_error():
    """When LLM returns non-JSON, agent should return a structured error dict."""
    from unittest.mock import MagicMock
    mock_msg = MagicMock()
    mock_msg.content = "I cannot classify this feedback."

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        result = await agent._run_classification(
            "Some feedback text.",
            context_id="test-006"
        )

    assert "error" in result
    assert result["sentiment"] is None
