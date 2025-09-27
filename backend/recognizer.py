"""Face recognition utilities that leverage Gemini Vision and local models."""
from __future__ import annotations

import hashlib
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .ai import GeminiService, GeminiUnavailableError
from .face_pipeline import AdvancedFacePipeline, AdvancedFacePipelineError

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
        gemini: GeminiService | None = None,
        tolerance: float = 0.32,
        arcface_tolerance: float = 0.35,
        pipeline: AdvancedFacePipeline | None = None,
    ) -> None:
        self._gemini = gemini
        self.tolerance = tolerance
        self.arcface_tolerance = arcface_tolerance
        self._pipeline = pipeline
        self._faiss = self._import_faiss()
        self._last_pipeline_result: AdvancedFacePipeline.PipelineResult | None = None

    def encode(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> FaceEncoding:
        vector: np.ndarray | None = None
        signature = ""
        source = "hash"

        self._last_pipeline_result = None

        if self._pipeline and self._pipeline.is_available:
            try:
                pipeline_result = self._pipeline.process_all(image_bytes, mime_type=mime_type)
            except AdvancedFacePipelineError as exc:
                _LOGGER.warning("Advanced face pipeline unavailable, falling back: %s", exc)
            else:
                self._last_pipeline_result = pipeline_result
                best_face = pipeline_result.faces[0]
                vector = best_face.embedding.astype(np.float32, copy=False)
                signature = self._hash_signature(vector.tobytes())
                source = "insightface"

        if self._gemini and self._gemini.can_describe_faces:
            try:
                gemini_signature = self._gemini.describe_face(image_bytes, mime_type)
                if gemini_signature:
                    if source == "hash":
                        source = "gemini"
                    else:
                        source = f"{source}+gemini"
                    signature = gemini_signature
            except GeminiUnavailableError as exc:
                _LOGGER.warning("Gemini Vision unavailable, falling back to local pipeline: %s", exc)

        if vector is None:
            if not signature:
                signature = self._hash_signature(image_bytes)
                source = "hash"
            vector = self._vector_from_signature(signature)
        else:
            if not signature:
                signature = self._hash_signature(vector.tobytes())

        return FaceEncoding(vector=vector.astype(np.float32, copy=False), signature=signature, source=source)

    # ------------------------------------------------------------------
    def consume_last_pipeline_result(self) -> AdvancedFacePipeline.PipelineResult | None:
        result = self._last_pipeline_result
        self._last_pipeline_result = None
        return result

    # ------------------------------------------------------------------
    def derive_member_id(self, encoding: FaceEncoding) -> str:
        base = encoding.signature or self._hash_signature(encoding.vector.tobytes())
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest().upper()
        return f"MEM{digest[:10]}"

    def distance(self, a: FaceEncoding, b: FaceEncoding) -> float:
        if a.vector.shape != b.vector.shape:
            return float("inf")
        if a.vector.size >= 256:
            return float(self._cosine_distance(a.vector, b.vector))
        return float(np.linalg.norm(a.vector - b.vector))

    def is_match(self, known: FaceEncoding, candidate: FaceEncoding) -> bool:
        if known.signature and candidate.signature and known.signature == candidate.signature:
            return True
        if known.vector.size == 0 or candidate.vector.size == 0:
            return False
        if known.vector.shape != candidate.vector.shape:
            return False
        distance = self.distance(known, candidate)
        threshold = self.arcface_tolerance if known.vector.size >= 256 else self.tolerance
        return distance <= threshold

    # ------------------------------------------------------------------
    def find_best_match(
        self,
        candidates: Sequence[tuple[str, FaceEncoding]],
        target: FaceEncoding,
    ) -> tuple[str | None, float | None]:
        if not candidates:
            return None, None

        best_member: str | None = None
        best_distance: float | None = None
        closest_distance: float | None = None

        if (
            self._faiss is not None
            and target.vector.size > 0
            and target.vector.ndim == 1
        ):
            faiss_vectors: list[np.ndarray] = []
            faiss_mapping: list[int] = []
            for idx, (member_id, encoding) in enumerate(candidates):
                if encoding.vector.shape == target.vector.shape and encoding.vector.size > 0:
                    faiss_vectors.append(encoding.vector.astype(np.float32, copy=False))
                    faiss_mapping.append(idx)
            if faiss_vectors:
                try:
                    matrix = np.stack(faiss_vectors).astype(np.float32, copy=False)
                    index = self._faiss.IndexFlatL2(matrix.shape[1])
                    index.add(matrix)
                    query = target.vector.reshape(1, -1).astype(np.float32, copy=False)
                    k = min(5, matrix.shape[0])
                    distances, indices = index.search(query, k)
                    for pos in indices[0]:
                        if pos < 0:
                            continue
                        candidate_idx = faiss_mapping[pos]
                        member_id, encoding = candidates[candidate_idx]
                        distance = self.distance(encoding, target)
                        if not np.isfinite(distance):
                            continue
                        if closest_distance is None or distance < closest_distance:
                            closest_distance = distance
                        match = self.is_match(encoding, target)
                        _LOGGER.debug(
                            "FAISS 比對 member=%s 距離=%.4f match=%s",
                            member_id,
                            distance,
                            match,
                        )
                        if match:
                            return member_id, distance
                except Exception as exc:
                    _LOGGER.warning("FAISS search failed, using linear scan: %s", exc)

        for member_id, encoding in candidates:
            distance = self.distance(encoding, target)
            if not np.isfinite(distance):
                continue
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
            match = self.is_match(encoding, target)
            _LOGGER.debug(
                "線性掃描比對 member=%s 距離=%.4f match=%s",
                member_id,
                distance,
                match,
            )
            if match:
                if best_distance is None or distance < best_distance:
                    best_member, best_distance = member_id, distance

        if best_member is not None:
            _LOGGER.debug(
                "選定既有會員 %s (距離 %.4f)",
                best_member,
                best_distance if best_distance is not None else float("nan"),
            )
            return best_member, best_distance

        _LOGGER.debug(
            "未找到符合門檻的會員，最近距離=%.4f",
            closest_distance if closest_distance is not None else float("nan"),
        )
        return None, closest_distance

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

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return float("inf")
        similarity = float(np.dot(a, b) / denom)
        similarity = max(min(similarity, 1.0), -1.0)
        return 1.0 - similarity

    @staticmethod
    def _import_faiss():
        for module_name in ("faiss", "faiss_cpu"):
            try:
                return importlib.import_module(module_name)
            except ImportError:  # pragma: no cover - optional dependency
                continue
        return None
