"""Face recognition utilities that leverage Amazon Rekognition for cloud IDs."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .aws import (
    FaceMatch,
    IndexedFace,
    NoFaceDetectedError,
    RekognitionService,
    RekognitionUnavailableError,
)

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

    def __init__(
        self,
        rekognition: RekognitionService | None = None,
        tolerance: float = 0.32,
    ) -> None:
        self._rekognition = rekognition
        self.tolerance = tolerance

    def encode(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> FaceEncoding:
        signature = ""
        source = "hash"
        match: FaceMatch | None = None
        summary = None
        no_face_detected = False
        if self._rekognition:
            try:
                match = self._rekognition.search_face(image_bytes)
            except RekognitionUnavailableError as exc:
                _LOGGER.info(
                    "Amazon Rekognition search unavailable, will fall back to detection: %s",
                    exc,
                )
            if match is None and self._rekognition.can_describe_faces:
                try:
                    summary = self._rekognition.describe_face(image_bytes)
                except NoFaceDetectedError as exc:
                    no_face_detected = True
                    _LOGGER.info("Amazon Rekognition 未在影像偵測到人臉: %s", exc)
                except RekognitionUnavailableError as exc:
                    _LOGGER.warning(
                        "Amazon Rekognition detect_faces unavailable, falling back to hash: %s",
                        exc,
                    )

        if match is None and summary is None and no_face_detected:
            raise ValueError("影像中未偵測到人臉，請重新拍攝")

        if match is not None:
            signature = match.to_signature()
            source = "rekognition-search"
        elif summary is not None:
            signature = summary.to_signature()
            source = "rekognition-detect"

        if not signature:
            signature = self._hash_signature(image_bytes)
            source = "hash"

        vector = self._vector_from_signature(signature)
        return FaceEncoding(vector=vector, signature=signature, source=source)

    def register_face(self, image_bytes: bytes, member_id: str) -> FaceEncoding | None:
        """Index a member face into Rekognition and return the cloud-backed encoding."""

        if not self._rekognition:
            return None

        try:
            indexed: IndexedFace = self._rekognition.index_face(
                image_bytes, external_image_id=member_id
            )
        except RekognitionUnavailableError as exc:
            _LOGGER.warning("Amazon Rekognition index_faces failed: %s", exc)
            return None

        signature = indexed.to_signature()
        vector = self._vector_from_signature(signature)
        return FaceEncoding(vector=vector, signature=signature, source="rekognition-index")

    def remove_member_faces(self, member_id: str) -> int:
        """Remove any Rekognition faces associated with ``member_id``."""

        if not self._rekognition:
            return 0

        try:
            return self._rekognition.remove_faces_by_external_ids([member_id])
        except RekognitionUnavailableError as exc:
            _LOGGER.warning("Amazon Rekognition delete_faces failed: %s", exc)
            return 0

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
