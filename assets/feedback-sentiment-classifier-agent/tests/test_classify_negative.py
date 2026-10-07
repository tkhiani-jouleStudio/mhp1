"""Unit test: negative feedback classification."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_classify_negative():
    """Negative feedback should return sentiment='negative'."""
    mock_response = json.dumps({
        "sentiment": "negative",
        "confidence": 0.92,
        "reasoning": "The feedback explicitly states disappointment with late delivery and damaged packaging.",
        "low_confidence": False,
    })
    mock_msg = MagicMock()
    mock_msg.content = mock_response

    from unittest.mock import patch
    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        result = await agent._run_classification(
            "The delivery was late and packaging was damaged. Very disappointed.",
            context_id="test-002"
        )

    assert result["sentiment"] == "negative"
    assert result["confidence"] > 0
    assert result["reasoning"]
    assert result["low_confidence"] is False
    assert "error" not in result
