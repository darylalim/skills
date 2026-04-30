"""Pre-conversion test for gradio-multimodal-fixture.

After mlx-app-converter, this file should patch both mlx_lm.load (LLM path)
and mlx_vlm.load (VLM path) independently. The VLM mock should set
mock_result.text on the mlx_vlm.generate return value to match the actual
GenerationResult dataclass shape.
"""
from unittest.mock import MagicMock, patch

from app import describe_image, follow_up


@patch("app.AutoProcessor.from_pretrained")
@patch("app.AutoModelForVision2Seq.from_pretrained")
def test_describe_image_returns_string(mock_model_cls, mock_processor_cls):
    mock_processor = MagicMock()
    mock_processor.decode.return_value = "a cat"
    mock_processor_cls.return_value = mock_processor
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    mock_model_cls.return_value = mock_model

    result = describe_image("Describe.", "img.jpg", mock_model, mock_processor)

    assert result == "a cat"


@patch("app.AutoTokenizer.from_pretrained")
@patch("app.AutoModelForCausalLM.from_pretrained")
def test_follow_up_returns_string(mock_model_cls, mock_tokenizer_cls):
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "thinking about life"
    mock_tokenizer_cls.return_value = mock_tokenizer
    mock_model = MagicMock()
    mock_model.generate.return_value = [[4, 5, 6]]
    mock_model_cls.return_value = mock_model

    result = follow_up("What might the cat be thinking?", mock_model, mock_tokenizer)

    assert result == "thinking about life"
