"""Face recognition utilities that leverage Azure Face for cloud IDs."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ai import AzureFaceError, AzureFaceService, FaceAnalysis

_LOGGER = logging.getLogger(__name__)


@dataclass
class FaceEncoding:
    """Container for the anonymised face signature."""

    vector: np.ndarray
    face_description: str
    source: str = "hash"
    azure_person_id: str | None = None
    azure_person_name: str | None = None
    azure_confidence: float | None = None

    def to_jsonable(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "vector": self.vector.astype(float).tolist(),
            "face_description": self.face_description,
            "source": self.source,
        }
        if self.azure_person_id:
            payload["azure_person_id"] = self.azure_person_id
        if self.azure_person_name:
            payload["azure_person_name"] = self.azure_person_name
        if self.azure_confidence is not None:
            payload["azure_confidence"] = float(self.azure_confidence)
        return payload

    @classmethod
    def from_jsonable(cls, data: Any) -> "FaceEncoding":
        if isinstance(data, list):  # legacy format (vector only)
            vector = np.asarray(data, dtype=np.float32)
            return cls(vector=vector, face_description="", source="legacy")
        if isinstance(data, dict):
            vector = np.asarray(data.get("vector", []), dtype=np.float32)
            face_description = str(
                data.get("face_description")
                or data.get("gemini_description")
                or data.get("signature", "")
            )
            source = str(data.get("source", "")) or "hash"
            azure_person_id = data.get("azure_person_id")
            azure_person_name = data.get("azure_person_name")
            azure_confidence = data.get("azure_confidence")
            try:
                confidence_value = float(azure_confidence)
            except (TypeError, ValueError):
                confidence_value = None
            return cls(
                vector=vector,
                face_description=face_description,
                source=source,
                azure_person_id=str(azure_person_id)
                if isinstance(azure_person_id, str) and azure_person_id
                else None,
                azure_person_name=str(azure_person_name)
                if isinstance(azure_person_name, str) and azure_person_name
                else None,
                azure_confidence=confidence_value,
            )
        raise TypeError(f"Unsupported encoding payload: {type(data)!r}")


class FaceRecognizer:
    """Encapsulates the logic used to anonymise and match members."""

    def __init__(self, azure_face: AzureFaceService | None = None, tolerance: float = 0.32) -> None:
        self._azure = azure_face
        self.tolerance = tolerance

    def encode(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> FaceEncoding:
        description = ""
        source = "hash"
        analysis: FaceAnalysis | None = None
        azure_person_id: str | None = None
        azure_person_name: str | None = None
        azure_confidence: float | None = None
        if self._azure and self._azure.can_describe_faces:
            try:
                analysis = self._azure.analyze_face(image_bytes, mime_type)
            except AzureFaceError as exc:
                _LOGGER.warning("Azure Face unavailable, falling back to hash: %s", exc)
            else:
                description = analysis.description
                azure_person_id = str(analysis.person_id) if analysis.person_id else None
                azure_person_name = str(analysis.person_name) if analysis.person_name else None
                azure_confidence = analysis.confidence
                if azure_person_id:
                    source = "azure-person-group"
                elif description:
                    source = "azure"

        signature_seed = azure_person_id or description
        if not signature_seed:
            signature_seed = self._hash_signature(image_bytes)
            source = "hash"

        vector = self._vector_from_signature(signature_seed)
        face_description = description or azure_person_name or signature_seed
        return FaceEncoding(
            vector=vector,
            face_description=face_description,
            source=source,
            azure_person_id=azure_person_id,
            azure_person_name=azure_person_name,
            azure_confidence=azure_confidence,
        )

    # ------------------------------------------------------------------
    def derive_member_id(self, encoding: FaceEncoding) -> str:
        base = (
            encoding.azure_person_name
            or encoding.azure_person_id
            or encoding.face_description
            or self._hash_signature(encoding.vector.tobytes())
        )
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest().upper()
        return f"MEM{digest[:10]}"

    def distance(self, a: FaceEncoding, b: FaceEncoding) -> float:
        if a.vector.shape != b.vector.shape:
            return float("inf")
        return float(np.linalg.norm(a.vector - b.vector))

    def is_match(self, known: FaceEncoding, candidate: FaceEncoding) -> bool:
        if known.azure_person_id and candidate.azure_person_id:
            return known.azure_person_id == candidate.azure_person_id
        if known.azure_person_name and candidate.azure_person_name:
            return known.azure_person_name == candidate.azure_person_name
        if known.face_description and candidate.face_description:
            return known.face_description == candidate.face_description
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

