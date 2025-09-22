"""Google Cloud Vision helpers for face detection signatures."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable

try:  # pragma: no cover - optional dependency for local/offline development
    from google.cloud import vision  # type: ignore
    from google.oauth2 import service_account  # type: ignore
except Exception:  # pragma: no cover - guard when the dependency is missing
    vision = None  # type: ignore
    service_account = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


class VisionUnavailableError(RuntimeError):
    """Raised when Google Vision features are requested but unavailable."""


@dataclass
class VisionFace:
    """Small container with the raw Vision face annotation and derived signature."""

    signature: str
    landmarks: dict[str, tuple[float, float, float]]
    bounding_box: tuple[tuple[float, float], ...]


class VisionService:
    """Thin wrapper around Google Cloud Vision face detection."""

    def __init__(
        self,
        *,
        credentials_json: str | None = None,
        credentials_path: str | None = None,
        max_faces: int = 1,
    ) -> None:
        self._max_faces = max_faces
        self._client = None

        if vision is None:
            _LOGGER.warning(
                "google-cloud-vision is not installed. Install it to enable Vision features."
            )
            return

        credentials_json = credentials_json or os.getenv("VISION_CREDENTIALS_JSON")
        credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        try:
            if credentials_json:
                if service_account is None:
                    raise VisionUnavailableError(
                        "google-auth is required to use inline credentials"
                    )
                info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(info)
                self._client = vision.ImageAnnotatorClient(credentials=credentials)
            elif credentials_path:
                self._client = vision.ImageAnnotatorClient.from_service_account_file(
                    credentials_path
                )
            else:
                self._client = vision.ImageAnnotatorClient()
            _LOGGER.info("Google Vision service initialised for face detection")
        except Exception as exc:  # pragma: no cover - runtime configuration failure
            _LOGGER.error("Failed to initialise Google Vision client: %s", exc)
            self._client = None

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    def describe_face(self, image_bytes: bytes) -> VisionFace:
        """Return a deterministic signature derived from the first detected face."""

        if not self.enabled:
            raise VisionUnavailableError("Vision client is not configured")

        assert self._client is not None  # for mypy
        image = vision.Image(content=image_bytes)
        response = self._client.face_detection(image=image, max_results=self._max_faces)
        if response.error.message:
            raise VisionUnavailableError(
                f"Vision API error: {response.error.message} ({response.error.code})"
            )
        if not response.face_annotations:
            raise ValueError("Google Vision 沒有偵測到人臉，請再試一次")

        face = response.face_annotations[0]
        signature = self._signature_from_face(face)
        landmarks = self._landmarks_from_face(face.landmarks)
        bounding = self._bounding_box(face.bounding_poly)
        return VisionFace(signature=signature, landmarks=landmarks, bounding_box=bounding)

    # ------------------------------------------------------------------
    @staticmethod
    def _signature_from_face(face: "vision.FaceAnnotation") -> str:
        """Condense Vision face attributes into a deterministic signature string."""

        parts: list[str] = []
        # Core likelihood scores and confidence (values are floats or enums -> ints).
        parts.append(f"conf:{face.detection_confidence:.4f}")
        parts.append(f"joy:{int(face.joy_likelihood)}")
        parts.append(f"sorrow:{int(face.sorrow_likelihood)}")
        parts.append(f"anger:{int(face.anger_likelihood)}")
        parts.append(f"surprise:{int(face.surprise_likelihood)}")
        parts.append(f"blurred:{int(face.blurred_likelihood)}")
        parts.append(f"headwear:{int(face.headwear_likelihood)}")

        def _iter_vertices(poly: "vision.BoundingPoly | None") -> Iterable[tuple[int, int]]:
            if not poly:
                return []
            vertices = getattr(poly, "vertices", []) or []
            return [
                (
                    int(getattr(vertex, "x", 0) or 0),
                    int(getattr(vertex, "y", 0) or 0),
                )
                for vertex in vertices
            ]

        for idx, (x, y) in enumerate(_iter_vertices(face.bounding_poly)):
            parts.append(f"bp{idx}:{x}:{y}")

        # Landmark positions provide a more detailed, but anonymised signature.
        for landmark in sorted(face.landmarks, key=self._landmark_label):
            lm_type = self._landmark_label(landmark)
            position = getattr(landmark, "position", None)
            x = getattr(position, "x", 0.0) if position else 0.0
            y = getattr(position, "y", 0.0) if position else 0.0
            z = getattr(position, "z", 0.0) if position else 0.0
            parts.append(f"lm:{lm_type}:{x:.2f}:{y:.2f}:{z:.2f}")

        signature = "|".join(parts)
        _LOGGER.debug("Vision signature: %s", signature)
        return signature

    @staticmethod
    def _landmarks_from_face(
        landmarks: Iterable["vision.FaceAnnotation.Landmark"],
    ) -> dict[str, tuple[float, float, float]]:
        payload: dict[str, tuple[float, float, float]] = {}
        for landmark in landmarks:
            lm_type = self._landmark_label(landmark)
            position = getattr(landmark, "position", None)
            x = float(getattr(position, "x", 0.0) or 0.0)
            y = float(getattr(position, "y", 0.0) or 0.0)
            z = float(getattr(position, "z", 0.0) or 0.0)
            payload[lm_type] = (round(x, 3), round(y, 3), round(z, 3))
        return payload

    @staticmethod
    def _bounding_box(
        bounding_poly: "vision.BoundingPoly | None",
    ) -> tuple[tuple[float, float], ...]:
        if not bounding_poly:
            return tuple()
        box: list[tuple[float, float]] = []
        for vertex in getattr(bounding_poly, "vertices", []) or []:
            x = float(getattr(vertex, "x", 0.0) or 0.0)
            y = float(getattr(vertex, "y", 0.0) or 0.0)
            box.append((round(x, 2), round(y, 2)))
        return tuple(box)

    @staticmethod
    def _landmark_label(landmark: "vision.FaceAnnotation.Landmark") -> str:
        lm_type = getattr(landmark, "type_", None)
        name = getattr(lm_type, "name", None)
        if name:
            return str(name)
        value = getattr(lm_type, "value", None)
        if value is not None:
            return str(value)
        try:
            return str(int(lm_type))
        except Exception:  # pragma: no cover - defensive fallback
            return str(lm_type)

