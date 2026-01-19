from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class LocalEmbedder:
    model_path: str = os.path.join(os.getenv("AI_DOMAIN_MODELS_DIR", "embeddings_models"), "rubert-mini-frida")
    device: Optional[str] = None
    batch_size: int = 32
    normalize: bool = True

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required to build local embeddings"
            ) from exc

        self._model = SentenceTransformer(self.model_path, device=self.device)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, 0), dtype="float32")
        vectors = self._model.encode(
            text_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        return vectors.astype("float32")

    def embed_file(self, path: str) -> np.ndarray:
        from ai_domain.rag.file_loader import load_text_from_file

        text = load_text_from_file(path)
        return self.embed_texts([text])
