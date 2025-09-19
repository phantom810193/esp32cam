"""Face recognition utilities for the ESP32-CAM MVP backend."""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency
    import face_recognition  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade when dlib is missing
    face_recognition = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


@dataclass
class FaceEncoding:
    """Simple wrapper for an encoding vector.

    Storing the data in a dataclass makes it easy to (de-)serialise to and from the
    SQLite database while keeping the recognition specific logic inside this module.
    """

    vector: np.ndarray

    def to_jsonable(self) -> list[float]:
        return self.vector.astype(float).tolist()

    @classmethod
    def from_jsonable(cls, data: list[float]) -> "FaceEncoding":
        return cls(np.asarray(data, dtype=np.float32))


class FaceRecognizer:
    """Encapsulates face encoding logic with a graceful fallback.

    When the optional :mod:`face_recognition` dependency (which bundles dlib) is
    available we use it to generate true facial embeddings.  Otherwise, we fall back
    to a deterministic hash of the input image.  While the hash based approach does
    not provide real biometric guarantees it keeps the end-to-end flow working on
    devices where dlib is not installed, which is convenient for development and
    automated tests.
    """

    def __init__(self, tolerance: float = 0.45) -> None:
        self.tolerance = tolerance
        self._has_face_recognition = face_recognition is not None
        if self._has_face_recognition:
            _LOGGER.info("Using dlib/face_recognition backend for embeddings")
        else:
            _LOGGER.warning(
                "face_recognition (dlib) is not available. Falling back to a hash-based"
                " pseudo encoder. Install face_recognition for production deployments."
            )

    def encode(self, image_bytes: bytes) -> FaceEncoding:
        if self._has_face_recognition:
            return self._encode_with_face_recognition(image_bytes)
        return self._encode_with_hash(image_bytes)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _encode_with_face_recognition(self, image_bytes: bytes) -> FaceEncoding:
        assert face_recognition is not None  # for the type-checker
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            raise ValueError("No faces detected in the uploaded image")
        return FaceEncoding(np.asarray(encodings[0], dtype=np.float32))

    def _encode_with_hash(self, image_bytes: bytes) -> FaceEncoding:
        digest = hashlib.sha256(image_bytes).digest()
        # Repeat the digest to create a 128-dim pseudo embedding similar to dlib's size.
        repeated = (digest * ((128 // len(digest)) + 1))[:128]
        vector = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
        # Normalise to [0, 1] to roughly match the magnitude of a real encoding.
        vector /= 255.0
        return FaceEncoding(vector)

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------
    def distance(self, a: FaceEncoding, b: FaceEncoding) -> float:
        return float(np.linalg.norm(a.vector - b.vector))

    def is_match(self, known: FaceEncoding, candidate: FaceEncoding) -> bool:
        if self._has_face_recognition:
            return self.distance(known, candidate) <= self.tolerance
        # When running in hash mode we expect identical vectors.
        return bool(np.array_equal(known.vector, candidate.vector))

