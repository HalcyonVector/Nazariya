"""Multilingual sentence embeddings for article similarity (title + lede)."""

from __future__ import annotations

import numpy as np

# key -> (Hugging Face model id, text prefix). E5 models expect "query: " for symmetric tasks.
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: "),
    "e5-base": ("intfloat/multilingual-e5-base", "query: "),
    "bge-m3": ("BAAI/bge-m3", ""),
    "mpnet": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", ""),
}
LEDE_WORDS = 200
MAX_TOKENS = 256  # same cap for every model so the comparison is fair (and CPU-friendly)

_loaded: dict = {}


def doc_text(title: str, text: str | None, words: int = LEDE_WORDS) -> str:
    """What we embed: the headline plus the first ~200 words (events are defined by the lede)."""
    lede = " ".join((text or "").split()[:words])
    return f"{title}\n{lede}".strip()


def load(key: str, device: str | None = None):
    if device is None:
        import torch

        # GPU when present (local dev / labelling), CPU elsewhere (GitHub Actions, home task)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if (key, device) not in _loaded:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODELS[key][0], device=device)
        model.max_seq_length = MAX_TOKENS
        _loaded[(key, device)] = model
    return _loaded[(key, device)]


def unload() -> None:
    """Free cached models (and GPU memory) between models in a comparison run."""
    _loaded.clear()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def embed(key: str, texts: list[str], batch_size: int = 32, progress: bool = False,
          device: str | None = None) -> np.ndarray:
    """L2-normalised float32 embeddings, so cosine similarity is a dot product."""
    prefix = MODELS[key][1]
    vecs = load(key, device).encode(
        [prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=progress,
    )
    return vecs.astype(np.float32)
