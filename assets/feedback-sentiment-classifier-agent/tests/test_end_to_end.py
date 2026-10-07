"""Integration test: end-to-end agent flow via A2A invoke with mocked LLM."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_end_to_end_classification():
    """Full end-to-end: send feedback text via agent.invoke, assert structured JSON output."""
    mock_response = json.dumps({
        "sentiment": "positive",
        "confidence": 0.93,
        "reasoning": "The customer expresses clear satisfaction with both the product quality and service speed.",
        "low_confidence": False,
    })
    mock_msg = MagicMock()
    mock_msg.content = mock_response

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        response = await agent.invoke(
            query="Excellent quality and very fast shipping! Will definitely order again.",
            context_id="e2e-001"
        )

    assert response.status == "completed"

    # Parse the JSON content from the response
    result = json.loads(response.message)
    assert result["sentiment"] in ("positive", "negative", "neutral")
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["reasoning"], str)
    assert len(result["reasoning"]) > 0
    assert isinstance(result["low_confidence"], bool)
    assert "error" not in result


@pytest.mark.asyncio
async def test_end_to_end_empty_input_returns_error():
    """Sending empty feedback text should return a structured error response, not raise."""
    from agent import SampleAgent
    agent = SampleAgent()
    response = await agent.invoke(
        query="",
        context_id="e2e-002"
    )

    assert response.status == "completed"
    result = json.loads(response.message)
    assert "error" in result
    assert result["sentiment"] is None


@pytest.mark.asyncio
async def test_end_to_end_stream_returns_chunks():
    """stream() should yield at least two chunks: 'Processing...' and final result."""
    mock_response = json.dumps({
        "sentiment": "negative",
        "confidence": 0.88,
        "reasoning": "The customer is unhappy with the response time.",
        "low_confidence": False,
    })
    mock_msg = MagicMock()
    mock_msg.content = mock_response

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        chunks = []
        async for chunk in agent.stream(
            "Support took 3 days to respond. Not acceptable.",
            context_id="e2e-003"
        ):
            chunks.append(chunk)

    assert len(chunks) >= 2
    # First chunk is in-progress
    assert chunks[0]["is_task_complete"] is False
    # Last chunk is complete
    final = chunks[-1]
    assert final["is_task_complete"] is True
    result = json.loads(final["content"])
    assert result["sentiment"] == "negative"
