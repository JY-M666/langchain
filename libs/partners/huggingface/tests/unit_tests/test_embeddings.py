import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from langchain_huggingface import HuggingFaceEmbeddings


def _make_embeddings(monkeypatch, torch_module):
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = SimpleNamespace(
        stop_multi_process_pool=MagicMock()
    )
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", sentence_transformers
    )

    embeddings = HuggingFaceEmbeddings.model_construct()
    object.__setattr__(embeddings, "multi_process", True)
    object.__setattr__(embeddings, "_client", MagicMock())
    object.__getattribute__(embeddings, "_client").encode_multi_process.return_value = (
        SimpleNamespace(tolist=lambda: [[0.1, 0.2]])
    )
    return embeddings, object.__getattribute__(embeddings, "_client")


@pytest.mark.parametrize(
    ("device_type", "device_count"),
    [("xpu", 3), ("cuda", 2)],
)
def test_multi_process_uses_torch_accelerator_devices(
    monkeypatch, device_type, device_count
):
    accelerator = MagicMock()
    accelerator.is_available.return_value = True
    accelerator.current_accelerator.return_value = SimpleNamespace(type=device_type)
    accelerator.device_count.return_value = device_count
    torch_module = ModuleType("torch")
    torch_module.accelerator = accelerator

    embeddings, client = _make_embeddings(monkeypatch, torch_module)

    assert embeddings._embed(["hello"], {}) == [[0.1, 0.2]]
    client.start_multi_process_pool.assert_called_once_with(
        target_devices=[f"{device_type}:{index}" for index in range(device_count)]
    )


def test_multi_process_keeps_cpu_fallback_when_no_accelerator_is_available(
    monkeypatch,
):
    accelerator = MagicMock()
    accelerator.is_available.return_value = False
    torch_module = ModuleType("torch")
    torch_module.accelerator = accelerator

    embeddings, client = _make_embeddings(monkeypatch, torch_module)

    embeddings._embed(["hello"], {})

    client.start_multi_process_pool.assert_called_once_with()


def test_multi_process_keeps_default_for_older_torch(monkeypatch):
    torch_module = ModuleType("torch")

    embeddings, client = _make_embeddings(monkeypatch, torch_module)

    embeddings._embed(["hello"], {})

    client.start_multi_process_pool.assert_called_once_with()
