import os
import time
import numpy as np
from typing import List, Union


class AmharicEmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device: str = None):
        self.model_name = model_name
        self._model = None
        self._device = device

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"Loading embedding model '{self.model_name}' on device '{self._device}'...")
            t0 = time.time()
            self._model = SentenceTransformer(self.model_name, device=self._device)
            load_time = time.time() - t0
            print(f"Embedding model loaded in {load_time:.2f}s.")

    def encode(self, texts: Union[str, List[str]], is_query: bool = False, batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) into L2-normalized numpy embedding vectors.
        For e5 models, queries should be prefixed with 'query: ' and passages with 'passage: '.
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        # Handle E5 prefix convention if applicable
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            prefixed_texts = [f"{prefix}{t}" for t in texts]
        else:
            prefixed_texts = texts

        embeddings = self._model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


if __name__ == "__main__":
    embedder = AmharicEmbeddingModel()
    sample_texts = ["የባንክ ሂሳብ እንዴት መክፈት እችላለሁ?", "ATM ካርድ ከጠፋብኝ ምን ማድረግ አለብኝ?"]
    vecs = embedder.encode(sample_texts, is_query=True)
    print("Encoded vectors shape:", vecs.shape)
