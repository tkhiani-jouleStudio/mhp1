"""Unit test: preprocess_feedback hook is called before LLM, and overriding it changes input."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_preprocess_feedback_called():
    """preprocess_feedback should be called with the original text."""
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
        calls = []

        original_preprocess = agent.preprocess_feedback
        def tracking_preprocess(text):
            calls.append(text)
            return original_preprocess(text)

        agent.preprocess_feedback = tracking_preprocess
        await agent._run_classification("Great product!", context_id="test-007")

    assert len(calls) == 1
    assert calls[0] == "Great product!"


@pytest.mark.asyncio
async def test_preprocess_feedback_override_changes_input():
    """Overriding preprocess_feedback should change what is sent to the LLM."""
    mock_response = json.dumps({
        "sentiment": "positive",
        "confidence": 0.85,
        "reasoning": "Positive feedback.",
        "low_confidence": False,
    })
    mock_msg = MagicMock()
    mock_msg.content = mock_response

    with patch("agent.SampleAgent._invoke_with_fallback", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = {"messages": [mock_msg]}

        from agent import SampleAgent
        agent = SampleAgent()
        agent.preprocess_feedback = lambda text: "[TRANSFORMED] " + text

        await agent._run_classification("Good service!", context_id="test-008")

        # The query sent to LLM should contain the transformed text
        call_kwargs = mock_invoke.call_args
        query_sent = call_kwargs[1].get("query") or call_kwargs[0][2]
        assert "[TRANSFORMED]" in query_sent
