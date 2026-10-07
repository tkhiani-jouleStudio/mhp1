"""Unit test: positive feedback classification."""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_classify_positive():
    """Positive feedback should return sentiment='positive' with non-empty reasoning."""
    mock_response = json.dumps({
        "sentiment": "positive",
        "confidence": 0.95,
        "reasoning": "The feedback expresses satisfaction with the product quality and delivery.",
        "low_confidence": False,
    })

    mock_msg = MagicMock()
    mock_msg.content = mock_response

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        result = await agent._run_classification(
            "Great product! Really happy with the quality and fast delivery.",
            context_id="test-001"
        )

    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0
    assert result["reasoning"]
    assert result["low_confidence"] is False
    assert "error" not in result
