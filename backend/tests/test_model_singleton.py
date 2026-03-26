"""
Tests for embeddings/model.py — shared SentenceTransformer singleton.

The model is never actually loaded; SentenceTransformer is mocked.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure the singleton is reset before and after every test."""
    import embeddings.model as m
    m._model_instance = None
    yield
    m._model_instance = None


class TestGetEmbeddingModel:
    def test_returns_a_model_instance(self):
        from embeddings.model import get_embedding_model
        mock_model = MagicMock()
        with patch("embeddings.model.SentenceTransformer", return_value=mock_model):
            result = get_embedding_model()
        assert result is mock_model

    def test_loads_model_only_once(self):
        from embeddings.model import get_embedding_model
        with patch("embeddings.model.SentenceTransformer") as mock_cls:
            mock_cls.return_value = MagicMock()
            get_embedding_model()
            get_embedding_model()
            get_embedding_model()
        assert mock_cls.call_count == 1

    def test_returns_same_instance_on_repeated_calls(self):
        from embeddings.model import get_embedding_model
        mock_model = MagicMock()
        with patch("embeddings.model.SentenceTransformer", return_value=mock_model):
            first = get_embedding_model()
            second = get_embedding_model()
        assert first is second

    def test_uses_multilingual_model(self):
        from embeddings.model import get_embedding_model, EMBEDDING_MODEL
        with patch("embeddings.model.SentenceTransformer") as mock_cls, \
             patch("embeddings.model._detect_device", return_value="cpu"):
            mock_cls.return_value = MagicMock()
            get_embedding_model()
        args, kwargs = mock_cls.call_args
        assert args[0] == EMBEDDING_MODEL

    def test_passes_device_to_sentence_transformer(self):
        from embeddings.model import get_embedding_model
        with patch("embeddings.model.SentenceTransformer") as mock_cls, \
             patch("embeddings.model._detect_device", return_value="cpu"):
            mock_cls.return_value = MagicMock()
            get_embedding_model()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("device") == "cpu"

    def test_env_var_overrides_device(self, monkeypatch):
        import embeddings.model as m
        monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
        with patch("embeddings.model.SentenceTransformer") as mock_cls:
            mock_cls.return_value = MagicMock()
            m.get_embedding_model()
        _, kwargs = mock_cls.call_args
        assert kwargs.get("device") == "cpu"

    def test_model_name_is_multilingual_minilm(self):
        from embeddings.model import EMBEDDING_MODEL
        assert "multilingual" in EMBEDDING_MODEL
        assert "MiniLM" in EMBEDDING_MODEL

    def test_detect_device_returns_string(self):
        from embeddings.model import _detect_device
        device = _detect_device()
        assert device in ("cuda", "mps", "cpu")
