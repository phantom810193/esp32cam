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
    gemini_description: str | None = None

    def to_jsonable(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "vector": self.vector.astype(float).tolist(),
            "signature": self.signature,
            "source": self.source,
        }
        if self.gemini_description is not None:
            payload["gemini_description"] = self.gemini_description
        return payload

    @classmethod
    def from_jsonable(cls, data: Any) -> "FaceEncoding":
        if isinstance(data, list):  # legacy format (vector only)
            vector = np.asarray(data, dtype=np.float32)
            return cls(vector=vector, signature="", source="legacy")
        if isinstance(data, dict):
            vector = np.asarray(data.get("vector", []), dtype=np.float32)
            raw_signature = data.get("signature", "")
            signature = str(raw_signature) if raw_signature is not None else ""
            source = str(data.get("source", "")) or "hash"
            description = data.get("gemini_description")
            if description is not None:
                description = str(description) or None
            elif source == "gemini" and signature:
                # Legacy entries stored the Gemini description directly as the signature.
                description = signature
            return cls(
                vector=vector,
                signature=signature,
                source=source,
                gemini_description=description,
            )
        raise TypeError(f"Unsupported encoding payload: {type(data)!r}")


class FaceRecognizer:
    """Encapsulates the logic used to anonymise and match members."""

    def __init__(self, gemini: GeminiService | None = None, tolerance: float = 0.32) -> None:
        self._gemini = gemini
        self.tolerance = tolerance

    def encode(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> FaceEncoding:
        description: str | None = None
        if self._gemini and self._gemini.can_describe_faces:
            try:
                description = (self._gemini.describe_face(image_bytes, mime_type) or "").strip()
            except GeminiUnavailableError as exc:
                _LOGGER.warning("Gemini Vision unavailable, falling back to hash metadata: %s", exc)
                description = None

        signature = self._hash_signature(image_bytes)
        source = "hash+gemini" if description else "hash"

        vector = self._vector_from_signature(signature)
        return FaceEncoding(
            vector=vector,
            signature=signature,
            source=source,
            gemini_description=description,
        )

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
            if known.signature == candidate.signature:
                return True
        if (
            known.gemini_description
            and candidate.gemini_description
            and known.gemini_description == candidate.gemini_description
        ):
            return True
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

