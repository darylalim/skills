"""Pre-conversion test for gradio-vlm-fixture."""
from unittest.mock import MagicMock, patch

from app import run_inference


@patch("app.AutoProcessor.from_pretrained")
@patch("app.AutoModelForVision2Seq.from_pretrained")
def test_run_inference_returns_string(mock_model_cls, mock_processor_cls):
    mock_processor = MagicMock()
    mock_processor.decode.return_value = "a cat sitting on a chair"
    mock_processor_cls.return_value = mock_processor
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    mock_model_cls.return_value = mock_model

    result = run_inference("Describe this image.", "fake.jpg", mock_model, mock_processor)

    assert isinstance(result, str)
    assert result == "a cat sitting on a chair"
