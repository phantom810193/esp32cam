"""Face recognition utilities backed by Google Vision or hashing fallbacks."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .vision import VisionService, VisionUnavailableError

_LOGGER = logging.getLogger(__name__)


@dataclass
class FaceEncoding:
    """Container for the anonymised face signature."""

    vector: np.ndarray
    signature: str
    source: str = "hash"
    metadata: dict[str, object] | None = None

    def to_jsonable(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "vector": self.vector.astype(float).tolist(),
            "signature": self.signature,
            "source": self.source,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_jsonable(cls, data: Any) -> "FaceEncoding":
        if isinstance(data, list):  # legacy format (vector only)
            vector = np.asarray(data, dtype=np.float32)
            return cls(vector=vector, signature="", source="legacy")
        if isinstance(data, dict):
            vector = np.asarray(data.get("vector", []), dtype=np.float32)
            signature = str(data.get("signature", ""))
            source = str(data.get("source", "")) or "hash"
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else None
            return cls(vector=vector, signature=signature, source=source, metadata=metadata)
        raise TypeError(f"Unsupported encoding payload: {type(data)!r}")


class FaceRecognizer:
    """Encapsulates the logic used to anonymise and match members."""

    def __init__(self, *, vision: VisionService | None = None, tolerance: float = 0.32) -> None:
        self._vision = vision
        self.tolerance = tolerance

    def encode(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> FaceEncoding:
        signature = ""
        source = "hash"
        metadata: dict[str, object] | None = None
        if self._vision and self._vision.enabled:
            try:
                face = self._vision.describe_face(image_bytes)
                signature = face.signature
                source = "vision"
                metadata = {
                    "vision": {
                        "landmarks": face.landmarks,
                        "bounding_box": face.bounding_box,
                    }
                }
            except VisionUnavailableError as exc:
                _LOGGER.warning("Vision service unavailable, falling back to hash: %s", exc)
            except ValueError:
                raise

        if not signature:
            signature = self._hash_signature(image_bytes)
            source = "hash"

        vector = self._vector_from_signature(signature)
        return FaceEncoding(vector=vector, signature=signature, source=source, metadata=metadata)

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

