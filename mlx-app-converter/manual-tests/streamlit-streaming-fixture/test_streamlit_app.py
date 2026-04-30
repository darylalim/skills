"""Pre-conversion test for streamlit-streaming-fixture.

Smoke-tests the streaming inference function. After mlx-app-converter is
attempted on this fixture, the conversion should be REJECTED (the skill
soft-rejects streaming sources). This file is therefore expected to remain
unchanged post-conversion-attempt.
"""
from unittest.mock import MagicMock, patch

from streamlit_app import run_inference


@patch("streamlit_app.Thread")
@patch("streamlit_app.TextIteratorStreamer")
@patch("streamlit_app.AutoTokenizer.from_pretrained")
@patch("streamlit_app.AutoModelForCausalLM.from_pretrained")
def test_run_inference_yields_strings(
    mock_model_cls,
    mock_tokenizer_cls,
    mock_streamer_cls,
    mock_thread_cls,
):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    mock_tokenizer_cls.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_model_cls.return_value = mock_model
    mock_streamer = MagicMock()
    mock_streamer.__iter__.return_value = iter(["hello", " world"])
    mock_streamer_cls.return_value = mock_streamer
    mock_thread = MagicMock()
    mock_thread_cls.return_value = mock_thread

    chunks = list(run_inference("hi", mock_model, mock_tokenizer))

    assert chunks == ["hello", " world"]
    mock_thread.start.assert_called_once()
