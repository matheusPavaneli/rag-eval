from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from rageval.retrieval.store import RetrievalError

MAX_TOKENS = 512


class CrossEncoder(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def score(self, question: str, texts: Sequence[str]) -> tuple[float, ...]: ...


class OnnxCrossEncoder:
    def __init__(self, model: str, revision: str, cache_dir: Path) -> None:
        self._model = model
        self._revision = revision
        self._cache_dir = cache_dir
        self._loaded: tuple[Tokenizer, Any, frozenset[str]] | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def revision(self) -> str:
        return self._revision

    def score(self, question: str, texts: Sequence[str]) -> tuple[float, ...]:
        if not texts:
            return ()
        tokenizer, session, input_names = self._load()

        try:
            encodings = tokenizer.encode_batch([(question, text) for text in texts])
            inputs = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            (logits,) = session.run(None, {k: v for k, v in inputs.items() if k in input_names})
        except Exception as error:
            raise RetrievalError(f"reranker {self._model} failed to score: {error}") from error

        return tuple(float(value) for value in np.asarray(logits).reshape(len(texts), -1)[:, 0])

    def _load(self) -> tuple[Tokenizer, Any, frozenset[str]]:
        if self._loaded is not None:
            return self._loaded

        try:
            tokenizer = Tokenizer.from_file(self._download("tokenizer.json"))
            tokenizer.enable_truncation(max_length=MAX_TOKENS)
            tokenizer.enable_padding()
            session = onnxruntime.InferenceSession(
                self._download("onnx/model.onnx"), providers=["CPUExecutionProvider"]
            )
        except RetrievalError:
            raise
        except Exception as error:
            raise RetrievalError(f"reranker {self._model} failed to load: {error}") from error

        names = frozenset(node.name for node in session.get_inputs())
        self._loaded = (tokenizer, session, names)
        return self._loaded

    def _download(self, filename: str) -> str:
        try:
            path = hf_hub_download(
                self._model,
                filename,
                revision=self._revision,
                cache_dir=self._cache_dir / "models",
            )
        except Exception as error:
            raise RetrievalError(
                f"reranker {self._model}@{self._revision} {filename} unavailable: {error}"
            ) from error
        return str(path)
