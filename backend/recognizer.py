"""Face recognition utilities that leverage Gemini Vision for cloud IDs."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ai import GeminiService, GeminiUnavailableError

_LOGGER = logging.getLogger(__name__)


@dataclass
class FaceEncoding:
    """Container for the anonymised face signature."""

    vector: np.ndarray
    signature: str
    source: str = "hash"

    def to_jsonable(self) -> dict[str, object]:
        return {
            "vector": self.vector.astype(float).tolist(),
            "signature": self.signature,
            "source": self.source,
        }

    @classmethod
    def from_jsonable(cls, data: Any) -> "FaceEncoding":
        if isinstance(data, list):  # legacy format (vector only)
            vector = np.asarray(data, dtype=np.float32)
            return cls(vector=vector, signature="", source="legacy")
        if isinstance(data, dict):
            vector = np.asarray(data.get("vector", []), dtype=np.float32)
            signature = str(data.get("signature", ""))
            source = str(data.get("source", "")) or "hash"
            return cls(vector=vector, signature=signature, source=source)
        raise TypeError(f"Unsupported encoding payload: {type(data)!r}")


class FaceRecognizer:
    """Encapsulates the logic used to anonymise and match members."""

    def __init__(self, gemini: GeminiService | None = None, tolerance: float = 0.32) -> None:
        self._gemini = gemini
        self.tolerance = tolerance

    def encode(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> FaceEncoding:
        signature = ""
        source = "hash"
        if self._gemini and self._gemini.can_describe_faces:
            try:
                signature = self._gemini.describe_face(image_bytes, mime_type)
                source = "gemini"
            except GeminiUnavailableError as exc:
                _LOGGER.warning("Gemini Vision unavailable, falling back to hash: %s", exc)

        if not signature:
            signature = self._hash_signature(image_bytes)
            source = "hash"

        vector = self._vector_from_signature(signature)
        return FaceEncoding(vector=vector, signature=signature, source=source)

    # ------------------------------------------------------------------
    def derive_member_id(self, encoding: FaceEncoding) -> str:
        base = encoding.signature or self._hash_signature(encoding.vector.tobytes())
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest().upper()
        return f"MEM{digest[:10]}"

    def distance(self, a: FaceEncoding, b: FaceEncoding) -> float:
        if a.vector.shape != b.vector.shape:
            return float("inf")
        return float(np.linalg.norm(a.vector - b.vector))

    def is_match(self, known: FaceEncoding, candidate: FaceEncoding) -> bool:
        if known.signature and candidate.signature:
            return known.signature == candidate.signature
        return self.distance(known, candidate) <= self.tolerance

    # ------------------------------------------------------------------
    @staticmethod
    def _hash_signature(payload: bytes | str) -> str:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _vector_from_signature(signature: str) -> np.ndarray:
        digest = hashlib.sha256(signature.encode("utf-8")).digest()
        repeated = (digest * ((128 // len(digest)) + 1))[:128]
        vector = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
        vector /= 255.0
        return vector

