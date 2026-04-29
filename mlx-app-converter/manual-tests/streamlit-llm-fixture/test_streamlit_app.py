"""Unit tests for streamlit_app — pre-conversion baseline (transformers-based)."""

from unittest.mock import MagicMock, patch

from streamlit_app import run_inference


@patch("streamlit_app.AutoModelForCausalLM.from_pretrained")
@patch("streamlit_app.AutoTokenizer.from_pretrained")
def test_run_inference_returns_string(mock_tokenizer_cls, mock_model_cls):
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "hello world"
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    mock_model_cls.return_value = mock_model

    result = run_inference("hi", mock_model, mock_tokenizer)

    assert isinstance(result, str)
