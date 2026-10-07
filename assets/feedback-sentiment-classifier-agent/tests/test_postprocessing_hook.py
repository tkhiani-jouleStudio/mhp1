"""Unit test: postprocess_result hook is called after LLM, and overriding it changes output."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_postprocess_result_called():
    """postprocess_result should be called with the LLM result dict."""
    mock_response = json.dumps({
        "sentiment": "positive",
        "confidence": 0.9,
        "reasoning": "Positive feedback detected.",
        "low_confidence": False,
    })
    mock_msg = MagicMock()
    mock_msg.content = mock_response

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        post_calls = []

        original = agent.postprocess_result
        def tracking_post(result):
            post_calls.append(result.copy())
            return original(result)

        agent.postprocess_result = tracking_post
        await agent._run_classification("Amazing experience!", context_id="test-009")

    assert len(post_calls) == 1
    assert "sentiment" in post_calls[0]


@pytest.mark.asyncio
async def test_postprocess_result_override_changes_output():
    """Overriding postprocess_result should change the final result."""
    mock_response = json.dumps({
        "sentiment": "positive",
        "confidence": 0.9,
        "reasoning": "Positive feedback detected.",
        "low_confidence": False,
    })
    mock_msg = MagicMock()
    mock_msg.content = mock_response

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        # Override to add a custom field
        agent.postprocess_result = lambda r: {**r, "custom_label": "GREAT"}

        result = await agent._run_classification("Amazing!", context_id="test-010")

    assert result.get("custom_label") == "GREAT"
    assert result["sentiment"] == "positive"
