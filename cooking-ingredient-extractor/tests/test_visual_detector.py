# tests/test_visual_detector.py
import pytest
from src.models.visual_detector import VisualDetector, DetectionConfig
from unittest.mock import Mock, patch

def test_detector_initialization():
    """Test visual detector initialization"""
    detector = VisualDetector()
    assert detector.config is not None
    assert detector.config.confidence_threshold == 0.5

@patch('torch.hub.load')
def test_model_loading(mock_load):
    """Test model loading"""
    mock_load.return_value = Mock()
    detector = VisualDetector()
    assert detector.model is not None