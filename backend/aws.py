"""Helpers for interacting with Amazon Rekognition."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

try:  # pragma: no cover - optional dependency for local development
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover - guard against missing dependencies
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore


_LOGGER = logging.getLogger(__name__)


class RekognitionUnavailableError(RuntimeError):
    """Raised when the Rekognition client cannot be used."""


class NoFaceDetectedError(RekognitionUnavailableError):
    """Raised when Rekognition processes an image without detecting a face."""


@dataclass
class FaceSummary:
    """Structured attributes extracted from Amazon Rekognition."""

    payload: dict[str, Any]

    def to_signature(self) -> str:
        """Return a deterministic JSON signature for the detected face."""

        return json.dumps(self.payload, ensure_ascii=False, sort_keys=True)


@dataclass
class FaceMatch:
    """Summary of a face match returned by Rekognition collections."""

    face_id: str
    external_image_id: str | None
    similarity: float
    confidence: float

    def to_signature(self) -> str:
        """Return the best identifier to use as a stable signature."""

        return self.external_image_id or self.face_id


@dataclass
class IndexedFace:
    """Details about a face indexed into a Rekognition collection."""

    face_id: str
    external_image_id: str | None
    confidence: float

    def to_signature(self) -> str:
        return self.external_image_id or self.face_id


class RekognitionService:
    """Wrapper around Amazon Rekognition used for detection and collections."""

    def __init__(
        self,
        *,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        collection_id: str | None = None,
    ) -> None:
        self.region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.collection_id = (
            collection_id
            or os.getenv("AWS_REKOGNITION_COLLECTION_ID")
            or "esp32cam-members"
        )
        self._client = None
        self._collection_ready = False
        if boto3 is None:
            _LOGGER.info("boto3 is not installed. Amazon Rekognition features disabled.")
            return

        try:
            session = boto3.session.Session(
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                aws_session_token=session_token,
                region_name=self.region_name,
            )
            self._client = session.client("rekognition")
            _LOGGER.info(
                "Amazon Rekognition client initialised for region %s", self.region_name
            )
            self.ensure_collection()
        except Exception as exc:  # pragma: no cover - configuration failure
            _LOGGER.error("Failed to create Amazon Rekognition client: %s", exc)
            self._client = None

    # ------------------------------------------------------------------
    @property
    def can_describe_faces(self) -> bool:
        return self._client is not None

    @property
    def can_search_faces(self) -> bool:
        return self._client is not None and self._collection_ready

    @property
    def can_index_faces(self) -> bool:
        return self.can_search_faces

    # ------------------------------------------------------------------
    def ensure_collection(self) -> bool:
        """Ensure the configured collection exists, creating it if necessary."""

        if self._client is None or not self.collection_id:
            return False

        assert self._client is not None  # type narrow for the analyser
        try:
            self._client.describe_collection(CollectionId=self.collection_id)
            self._collection_ready = True
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException":
                return self._create_collection()
            if code == "AccessDeniedException":
                _LOGGER.error(
                    "Amazon Rekognition cannot access collection %s: %s",
                    self.collection_id,
                    exc,
                )
            else:
                _LOGGER.error(
                    "Amazon Rekognition describe_collection failed: %s", exc
                )
        except BotoCoreError as exc:  # pragma: no cover
            _LOGGER.error(
                "Amazon Rekognition describe_collection transport failure: %s", exc
            )

        self._collection_ready = False
        return False

    def _create_collection(self) -> bool:
        assert self._client is not None
        try:
            self._client.create_collection(CollectionId=self.collection_id)
            _LOGGER.info(
                "Amazon Rekognition collection %s ensured", self.collection_id
            )
            self._collection_ready = True
            return True
        except ClientError as exc:  # pragma: no cover - API failure
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "ResourceAlreadyExistsException":
                self._collection_ready = True
                return True
            _LOGGER.error(
                "Amazon Rekognition create_collection %s failed: %s",
                self.collection_id,
                exc,
            )
        except BotoCoreError as exc:  # pragma: no cover - transport failure
            _LOGGER.error(
                "Amazon Rekognition create_collection transport failure: %s", exc
            )
        self._collection_ready = False
        return False

    # ------------------------------------------------------------------
    def describe_face(self, image_bytes: bytes) -> FaceSummary:
        """Detect a face and convert the response into a stable signature."""

        if not self.can_describe_faces:
            raise RekognitionUnavailableError("Amazon Rekognition client is not configured")

        assert self._client is not None  # for the type checker
        try:
            response = self._client.detect_faces(
                Image={"Bytes": image_bytes},
                Attributes=["ALL"],
            )
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - API failure
            raise RekognitionUnavailableError(f"Amazon Rekognition detect_faces failed: {exc}") from exc

        faces = response.get("FaceDetails", [])
        if not faces:
            raise NoFaceDetectedError("Amazon Rekognition 未在影像中偵測到人臉")

        signature_payload = self._build_face_payload(faces[0])
        _LOGGER.debug("Amazon Rekognition signature payload: %s", signature_payload)
        return FaceSummary(payload=signature_payload)

    # ------------------------------------------------------------------
    def search_face(
        self,
        image_bytes: bytes,
        *,
        threshold: float = 90.0,
        max_faces: int = 1,
    ) -> FaceMatch | None:
        """Search the configured collection for the closest matching face."""

        if not self.can_search_faces and not self.ensure_collection():
            raise RekognitionUnavailableError(
                "Amazon Rekognition collection is not available"
            )

        assert self._client is not None
        try:
            response = self._client.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={"Bytes": image_bytes},
                FaceMatchThreshold=threshold,
                MaxFaces=max_faces,
            )
        except ClientError as exc:  # pragma: no cover - API failure
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException":
                self._collection_ready = False
                raise RekognitionUnavailableError(
                    f"Amazon Rekognition collection {self.collection_id} was not found"
                ) from exc
            raise RekognitionUnavailableError(
                f"Amazon Rekognition search_faces_by_image failed: {exc}"
            ) from exc
        except BotoCoreError as exc:  # pragma: no cover - transport failure
            raise RekognitionUnavailableError(
                f"Amazon Rekognition search_faces_by_image transport failure: {exc}"
            ) from exc

        matches = response.get("FaceMatches", [])
        if not matches:
            return None

        best = matches[0]
        face = best.get("Face", {})
        return FaceMatch(
            face_id=str(face.get("FaceId", "")),
            external_image_id=(
                str(face.get("ExternalImageId")) if face.get("ExternalImageId") else None
            ),
            similarity=float(best.get("Similarity", 0.0)),
            confidence=float(face.get("Confidence", 0.0)),
        )

    # ------------------------------------------------------------------
    def index_face(
        self,
        image_bytes: bytes,
        *,
        external_image_id: str | None = None,
        detection_attributes: Iterable[str] | None = None,
    ) -> IndexedFace:
        """Store the provided face in the configured collection."""

        if not self.can_index_faces and not self.ensure_collection():
            raise RekognitionUnavailableError(
                "Amazon Rekognition collection is not available for indexing"
            )

        assert self._client is not None
        try:
            response = self._client.index_faces(
                CollectionId=self.collection_id,
                Image={"Bytes": image_bytes},
                ExternalImageId=external_image_id,
                DetectionAttributes=list(detection_attributes or []),
                MaxFaces=1,
                QualityFilter="AUTO",
            )
        except ClientError as exc:  # pragma: no cover - API failure
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException":
                self._collection_ready = False
                raise RekognitionUnavailableError(
                    f"Amazon Rekognition collection {self.collection_id} was not found"
                ) from exc
            raise RekognitionUnavailableError(
                f"Amazon Rekognition index_faces failed: {exc}"
            ) from exc
        except BotoCoreError as exc:  # pragma: no cover - transport failure
            raise RekognitionUnavailableError(
                f"Amazon Rekognition index_faces transport failure: {exc}"
            ) from exc

        records = response.get("FaceRecords", [])
        if not records:
            raise RekognitionUnavailableError("Amazon Rekognition 無法從影像建立人臉索引")

        face = records[0].get("Face", {})
        return IndexedFace(
            face_id=str(face.get("FaceId", "")),
            external_image_id=(
                str(face.get("ExternalImageId")) if face.get("ExternalImageId") else None
            ),
            confidence=float(face.get("Confidence", 0.0)),
        )

    # ------------------------------------------------------------------
    def remove_faces(self, face_ids: Sequence[str]) -> int:
        """Delete the provided face IDs from the configured collection."""

        if not face_ids:
            return 0

        if not self.can_index_faces and not self.ensure_collection():
            raise RekognitionUnavailableError(
                "Amazon Rekognition collection is not available for deletion"
            )

        assert self._client is not None
        try:
            response = self._client.delete_faces(
                CollectionId=self.collection_id,
                FaceIds=list(face_ids),
            )
        except ClientError as exc:  # pragma: no cover - API failure
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException":
                self._collection_ready = False
                raise RekognitionUnavailableError(
                    f"Amazon Rekognition collection {self.collection_id} was not found"
                ) from exc
            raise RekognitionUnavailableError(
                f"Amazon Rekognition delete_faces failed: {exc}"
            ) from exc
        except BotoCoreError as exc:  # pragma: no cover - transport failure
            raise RekognitionUnavailableError(
                f"Amazon Rekognition delete_faces transport failure: {exc}"
            ) from exc

        deleted = response.get("DeletedFaces", [])
        return len(deleted)

    def remove_faces_by_external_ids(self, external_ids: Iterable[str]) -> int:
        """Delete all faces whose ``ExternalImageId`` matches the provided IDs."""

        ids = {external_id for external_id in external_ids if external_id}
        if not ids:
            return 0

        matches: list[str] = []
        for face in self._iter_collection_faces():
            external_id = face.get("ExternalImageId")
            if external_id and external_id in ids:
                matches.append(str(face.get("FaceId", "")))

        if not matches:
            return 0

        return self.remove_faces(matches)

    def _iter_collection_faces(self) -> Iterator[dict[str, Any]]:
        """Yield faces stored in the collection, handling pagination transparently."""

        if not self.can_index_faces and not self.ensure_collection():
            raise RekognitionUnavailableError(
                "Amazon Rekognition collection is not available for listing"
            )

        assert self._client is not None
        pagination_token: str | None = None
        while True:
            try:
                params = {
                    "CollectionId": self.collection_id,
                    "MaxResults": 1000,
                }
                if pagination_token:
                    params["NextToken"] = pagination_token
                response = self._client.list_faces(**params)
            except ClientError as exc:  # pragma: no cover - API failure
                code = exc.response.get("Error", {}).get("Code", "")
                if code == "ResourceNotFoundException":
                    self._collection_ready = False
                    raise RekognitionUnavailableError(
                        f"Amazon Rekognition collection {self.collection_id} was not found"
                    ) from exc
                raise RekognitionUnavailableError(
                    f"Amazon Rekognition list_faces failed: {exc}"
                ) from exc
            except BotoCoreError as exc:  # pragma: no cover - transport failure
                raise RekognitionUnavailableError(
                    f"Amazon Rekognition list_faces transport failure: {exc}"
                ) from exc

            for face in response.get("Faces", []) or []:
                yield face

            pagination_token = response.get("NextToken")
            if not pagination_token:
                break

    # ------------------------------------------------------------------
    @staticmethod
    def _build_face_payload(face: dict[str, Any]) -> dict[str, Any]:
        """Transform the Rekognition response into a condensed payload."""

        def _round_map(source: dict[str, Any], keys: tuple[str, ...], precision: int) -> dict[str, float]:
            return {
                key.lower(): round(float(source.get(key, 0.0)), precision)
                for key in keys
                if key in source
            }

        bounding_box = _round_map(face.get("BoundingBox", {}), ("Width", "Height", "Left", "Top"), 4)
        pose = _round_map(face.get("Pose", {}), ("Roll", "Yaw", "Pitch"), 2)
        quality = _round_map(face.get("Quality", {}), ("Sharpness", "Brightness"), 2)

        landmarks = [
            {
                "type": str(point.get("Type", "")),
                "x": round(float(point.get("X", 0.0)), 3),
                "y": round(float(point.get("Y", 0.0)), 3),
            }
            for point in face.get("Landmarks", [])
        ]
        landmarks.sort(key=lambda item: item["type"])

        def _bool_value(attribute: str) -> bool:
            value = face.get(attribute, {}).get("Value")
            return bool(value)

        appearance_flags = {
            "beard": _bool_value("Beard"),
            "mustache": _bool_value("Mustache"),
            "eyeglasses": _bool_value("Eyeglasses"),
            "sunglasses": _bool_value("Sunglasses"),
            "eyes_open": _bool_value("EyesOpen"),
            "mouth_open": _bool_value("MouthOpen"),
            "smile": _bool_value("Smile"),
        }

        emotions = sorted(
            (
                {
                    "type": str(item.get("Type", "")),
                    "confidence": round(float(item.get("Confidence", 0.0)), 1),
                }
                for item in face.get("Emotions", [])
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )[:3]

        age_range = face.get("AgeRange", {})
        payload = {
            "bounding_box": bounding_box,
            "pose": pose,
            "quality": quality,
            "landmarks": landmarks,
            "appearance": appearance_flags,
            "emotions": emotions,
            "gender": str(face.get("Gender", {}).get("Value", "")).lower(),
            "age": {
                "low": age_range.get("Low"),
                "high": age_range.get("High"),
            },
        }
        return payload
