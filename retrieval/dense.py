"""Dense retrieval with a local sentence encoder.

The other half of the hybrid retriever, and the first place the GPU earns its
keep. A base-size encoder in fp16 is roughly 1-1.5 GB of weights plus a modest
activation footprint at sequence length 256 -- comfortably inside the 6 GB
floor with room for a real batch size.

Model choice, device, precision, batch size and sequence length all come from
``config/hardware.yaml``. Nothing here is hardcoded, so the same code runs on
an 8 GB card, a 6 GB card, and a CPU-only CI box.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import settings
from retrieval.corpus import Corpus
from retrieval.sparse import ScoredPassage


@dataclass
class DenseRetriever:
    """Bi-encoder retrieval over cached passage embeddings.

    Embeddings are computed once per corpus and reused. On a laptop GPU the
    encode pass dominates everything else in A1, so re-encoding on every query
    would make the dev loop painful for no reason.
    """

    corpus: Corpus
    profile: str | None = None
    _model: object | None = None
    _embeddings: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.hw = settings.hardware(self.profile)
        self.spec = self.hw.embeddings

    # -- model --------------------------------------------------------------- #

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError("pip install sentence-transformers") from exc

            device = self.hw.resolve_device()
            model = SentenceTransformer(self.spec.model, device=device)
            model.max_seq_length = self.spec.max_seq_length
            if self.hw.use_fp16():
                model = model.half()
            self._model = model
        return self._model

    # -- indexing ------------------------------------------------------------ #

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=self.spec.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # so dot product is cosine similarity
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def index(self, *, cache_path: str | Path | None = None) -> None:
        """Embed the corpus, optionally persisting to disk.

        The cache is keyed on corpus size and model name, so changing either
        invalidates it. That is coarse but correct: a stale embedding matrix
        silently retrieving the wrong passages is far worse than re-encoding.
        """
        if cache_path is not None:
            path = Path(cache_path)
            if path.exists():
                blob = np.load(path, allow_pickle=False)
                if (
                    blob["n"].item() == len(self.corpus)
                    and str(blob["model"].item()) == self.spec.model
                ):
                    self._embeddings = blob["emb"]
                    return

        self._embeddings = self.encode(self.corpus.texts)

        if cache_path is not None:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, emb=self._embeddings,
                n=np.array(len(self.corpus)), model=np.array(self.spec.model),
            )

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self.index()
        assert self._embeddings is not None
        return self._embeddings

    # -- search --------------------------------------------------------------- #

    def search(self, query: str, k: int = 5) -> list[ScoredPassage]:
        if len(self.corpus) == 0:
            return []
        q = self.encode([query])[0]
        scores = self.embeddings @ q
        ids = self.corpus.ids
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], ids[i]))
        return [
            ScoredPassage(passage_id=ids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order[:k], start=1)
        ]

    def embed_pairs(self, texts: list[str]) -> np.ndarray:
        """Expose embeddings for the A2 interaction-feature head.

        A2's learned head runs over ``[e_i; e_j; |e_i - e_j|; e_i * e_j]``, and
        reusing this encoder rather than loading a second one keeps the VRAM
        budget honest.
        """
        return self.encode(texts)
