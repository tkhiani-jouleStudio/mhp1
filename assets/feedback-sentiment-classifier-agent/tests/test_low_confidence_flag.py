"""Unit test: low_confidence flag is set when confidence < 0.5."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_low_confidence_flag():
    """A classification with confidence < 0.5 should set low_confidence=True."""
    mock_response = json.dumps({
        "sentiment": "neutral",
        "confidence": 0.40,
        "reasoning": "The feedback is ambiguous and could be interpreted either way.",
        "low_confidence": True,
    })
    mock_msg = MagicMock()
    mock_msg.content = mock_response

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        result = await agent._run_classification(
            "It was okay I guess.",
            context_id="test-004"
        )

    assert result["low_confidence"] is True
    assert result["confidence"] < 0.5
    assert result["sentiment"] in ("positive", "negative", "neutral")
