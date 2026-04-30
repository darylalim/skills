"""Pre-conversion test for streamlit-multimodal-fixture.

After mlx-app-converter, this file should patch both mlx_lm.load (LLM path)
and mlx_vlm.load (VLM path) independently.
"""
from unittest.mock import MagicMock, patch

from streamlit_app import describe_image, follow_up


@patch("streamlit_app.AutoProcessor.from_pretrained")
@patch("streamlit_app.AutoModelForVision2Seq.from_pretrained")
def test_describe_image_returns_string(mock_model_cls, mock_processor_cls):
    mock_processor = MagicMock()
    mock_processor.decode.return_value = "a cat"
    mock_processor_cls.return_value = mock_processor
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    mock_model_cls.return_value = mock_model

    result = describe_image("Describe.", "img.jpg", mock_model, mock_processor)

    assert result == "a cat"


@patch("streamlit_app.AutoTokenizer.from_pretrained")
@patch("streamlit_app.AutoModelForCausalLM.from_pretrained")
def test_follow_up_returns_string(mock_model_cls, mock_tokenizer_cls):
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "thinking about life"
    mock_tokenizer_cls.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_model.generate.return_value = [[4, 5, 6]]
    mock_model_cls.return_value = mock_model

    result = follow_up("What might the cat be thinking?", mock_model, mock_tokenizer)

    assert result == "thinking about life"
