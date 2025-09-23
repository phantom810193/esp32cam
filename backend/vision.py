"""Utilities to normalise Google Vision face annotations for the MVP backend."""
from __future__ import annotations

from typing import Any, Iterable


class VisionServiceError(RuntimeError):
    """Raised when the Vision API cannot be used."""


class VisionService:
    """Thin wrapper responsible for parsing Google Cloud Vision results."""

    _LANDMARK_OVERRIDES = {
        "LEFT_EYE_PUPIL": "left pupil",
        "RIGHT_EYE_PUPIL": "right pupil",
        "LEFT_EYEBROW_UPPER_MIDPOINT": "left eyebrow (upper)",
        "RIGHT_EYEBROW_UPPER_MIDPOINT": "right eyebrow (upper)",
        "LEFT_EAR_TRAGION": "left ear",
        "RIGHT_EAR_TRAGION": "right ear",
        "FOREHEAD_GLABELLA": "forehead",
        "CHIN_GNATHION": "chin",
        "CHIN_LEFT_GONION": "chin (left)",
        "CHIN_RIGHT_GONION": "chin (right)",
        "LEFT_CHEEK_CENTER": "left cheek",
        "RIGHT_CHEEK_CENTER": "right cheek",
    }

    _LIKELIHOOD_FIELDS = {
        "joy": "joy_likelihood",
        "sorrow": "sorrow_likelihood",
        "anger": "anger_likelihood",
        "surprise": "surprise_likelihood",
        "under_exposed": "under_exposed_likelihood",
        "blurred": "blurred_likelihood",
        "headwear": "headwear_likelihood",
    }

    _LIKELIHOOD_LABELS = {
        0: "unknown",
        1: "very unlikely",
        2: "unlikely",
        3: "possible",
        4: "likely",
        5: "very likely",
    }

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    # ------------------------------------------------------------------
    @staticmethod
    def _extract(obj: Any, attribute: str, default: Any | None = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attribute, default)
        return getattr(obj, attribute, default)

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _likelihood_label(value: Any) -> str:
        if value is None:
            return "unknown"
        if hasattr(value, "name"):
            value = value.name
        if isinstance(value, str):
            key = value.split(".")[-1].strip().upper()
            for number, label in VisionService._LIKELIHOOD_LABELS.items():
                if label.replace(" ", "").upper() == key:
                    return label
            # Handle Google enums such as "VERY_LIKELY"
            if key in ("UNKNOWN", "VERY_UNLIKELY", "UNLIKELY", "POSSIBLE", "LIKELY", "VERY_LIKELY"):
                mapping = {
                    "UNKNOWN": "unknown",
                    "VERY_UNLIKELY": "very unlikely",
                    "UNLIKELY": "unlikely",
                    "POSSIBLE": "possible",
                    "LIKELY": "likely",
                    "VERY_LIKELY": "very likely",
                }
                return mapping[key]
            try:
                value = int(value)
            except ValueError:
                return value.replace("_", " ").lower()
        if isinstance(value, int):
            return VisionService._LIKELIHOOD_LABELS.get(value, str(value))
        return str(value).replace("_", " ").lower()

    @staticmethod
    def _landmark_label(raw_type: Any) -> str:
        if raw_type is None:
            return "unknown"
        if hasattr(raw_type, "name"):
            key = raw_type.name
        else:
            key = str(raw_type)
        key = key.split(".")[-1].strip().upper()
        if key in VisionService._LANDMARK_OVERRIDES:
            return VisionService._LANDMARK_OVERRIDES[key]
        if key.endswith("_LANDMARK"):
            key = key[: -len("_LANDMARK")]
        return key.replace("_", " ").lower()

    @staticmethod
    def _position_to_dict(position: Any) -> dict[str, float | None]:
        if position is None:
            return {"x": None, "y": None, "z": None}
        if isinstance(position, dict):
            x = position.get("x")
            y = position.get("y")
            z = position.get("z")
        else:
            x = getattr(position, "x", None)
            y = getattr(position, "y", None)
            z = getattr(position, "z", None)
        return {
            "x": VisionService._float_or_none(x),
            "y": VisionService._float_or_none(y),
            "z": VisionService._float_or_none(z),
        }

    @staticmethod
    def _vertices_from_polygon(polygon: Any) -> list[dict[str, float | None]]:
        if polygon is None:
            return []
        vertices: Iterable[Any]
        if isinstance(polygon, dict):
            vertices = polygon.get("vertices", []) or []
        else:
            vertices = getattr(polygon, "vertices", []) or []
        results: list[dict[str, float | None]] = []
        for vertex in vertices:
            if vertex is None:
                continue
            if isinstance(vertex, dict):
                x = vertex.get("x")
                y = vertex.get("y")
            else:
                x = getattr(vertex, "x", None)
                y = getattr(vertex, "y", None)
            results.append({"x": VisionService._float_or_none(x), "y": VisionService._float_or_none(y)})
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def _landmarks_from_face(face: Any) -> list[dict[str, Any]]:
        landmarks = VisionService._extract(face, "landmarks", []) or []
        results: list[dict[str, Any]] = []
        for landmark in landmarks:
            if landmark is None:
                continue
            raw_type = None
            if isinstance(landmark, dict):
                raw_type = landmark.get("type") or landmark.get("type_")
                position = landmark.get("position")
                if position is None:
                    position = {
                        "x": landmark.get("x"),
                        "y": landmark.get("y"),
                        "z": landmark.get("z"),
                    }
            else:
                raw_type = getattr(landmark, "type_", None) or getattr(landmark, "type", None)
                position = getattr(landmark, "position", None)
                if position is None and hasattr(landmark, "x"):
                    position = {
                        "x": getattr(landmark, "x", None),
                        "y": getattr(landmark, "y", None),
                        "z": getattr(landmark, "z", None),
                    }
            label = VisionService._landmark_label(raw_type)
            results.append({"label": label, "position": VisionService._position_to_dict(position)})
        return results

    @staticmethod
    def _signature_from_face(face: Any) -> dict[str, Any]:
        signature: dict[str, Any] = {
            "detection_confidence": VisionService._float_or_none(
                VisionService._extract(face, "detection_confidence")
            ),
            "landmarking_confidence": VisionService._float_or_none(
                VisionService._extract(face, "landmarking_confidence")
            ),
            "angles": {
                "roll": VisionService._float_or_none(VisionService._extract(face, "roll_angle")),
                "pan": VisionService._float_or_none(VisionService._extract(face, "pan_angle")),
                "tilt": VisionService._float_or_none(VisionService._extract(face, "tilt_angle")),
            },
            "bounding_poly": VisionService._vertices_from_polygon(
                VisionService._extract(face, "bounding_poly")
            ),
            "fd_bounding_poly": VisionService._vertices_from_polygon(
                VisionService._extract(face, "fd_bounding_poly")
            ),
            "landmarks": VisionService._landmarks_from_face(face),
            "likelihoods": {
                field: VisionService._likelihood_label(VisionService._extract(face, attribute))
                for field, attribute in VisionService._LIKELIHOOD_FIELDS.items()
            },
        }
        return signature

