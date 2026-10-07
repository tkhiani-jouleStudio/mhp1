"""Unit test: neutral feedback classification."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_classify_neutral():
    """Neutral feedback should return sentiment='neutral'."""
    mock_response = json.dumps({
        "sentiment": "neutral",
        "confidence": 0.80,
        "reasoning": "The feedback describes the product without strong positive or negative language.",
        "low_confidence": False,
    })
    mock_msg = MagicMock()
    mock_msg.content = mock_response

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        result = await agent._run_classification(
            "The product arrived on time. It works as described.",
            context_id="test-003"
        )

    assert result["sentiment"] == "neutral"
    assert result["confidence"] > 0
    assert result["reasoning"]
    assert result["low_confidence"] is False
    assert "error" not in result
