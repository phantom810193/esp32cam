"""Utilities for interacting with Azure Face to power the cloud-based MVP."""
from __future__ import annotations

import inspect
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

try:  # pragma: no cover - optional dependency for offline development
    from azure.ai.vision.face import FaceClient as VisionFaceClient  # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore
except Exception:  # pragma: no cover - guard against missing dependencies
    VisionFaceClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore

try:  # pragma: no cover - optional dependency for offline development
    from azure.cognitiveservices.vision.face import FaceClient as CognitiveFaceClient  # type: ignore
    from azure.cognitiveservices.vision.face.models import FaceAttributeType  # type: ignore
    from msrest.authentication import CognitiveServicesCredentials  # type: ignore
except Exception:  # pragma: no cover - guard against missing dependencies
    CognitiveFaceClient = None  # type: ignore
    FaceAttributeType = None  # type: ignore
    CognitiveServicesCredentials = None  # type: ignore

_LOGGER = logging.getLogger(__name__)


class AzureFaceError(RuntimeError):
    """Raised when Azure Face features are requested but not configured."""


@dataclass
class AdCreative:
    """Structured ad copy returned by the AI generator."""

    headline: str
    subheading: str
    highlight: str


@dataclass
class FaceAnalysis:
    """Aggregated view of a detected face returned by Azure Face."""

    description: str
    face_id: str | None = None
    person_id: str | None = None
    person_name: str | None = None
    confidence: float | None = None


class AzureFaceService:
    """Thin wrapper around the Azure Face APIs used by the MVP."""

    _DEFAULT_FACE_ATTRIBUTES: Sequence[str] = (
        "age",
        "gender",
        "glasses",
        "facialHair",
        "emotion",
        "hair",
        "makeup",
        "accessories",
        "smile",
    )

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        *,
        client: Any | None = None,
        detection_model: str = "detection_04",
        recognition_model: str = "recognition_04",
        person_group_id: str | None = None,
    ) -> None:
        self.endpoint = endpoint or os.getenv("AZURE_FACE_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_FACE_KEY")
        self._detection_model = detection_model
        self._recognition_model = recognition_model
        self._client = client
        self._client_variant: str | None = None
        self._rest_adapter: _AzureFaceRestAdapter | None = None
        self.person_group_id = (
            (person_group_id or os.getenv("AZURE_FACE_PERSON_GROUP_ID") or "esp32cam-mvp")
            .strip()
            .lower()
        )
        self._person_group_ready = False

        if self._client is None and self.endpoint and self.api_key:
            self._client, self._client_variant = self._create_client()
        elif self._client is not None:
            self._client_variant = "custom"
        else:
            _LOGGER.info(
                "Azure Face 未設定（需要 AZURE_FACE_ENDPOINT / AZURE_FACE_KEY），將使用雜湊 fallback"
            )

        if self._client is not None and self.person_group_id:
            try:
                self._ensure_person_group()
            except AzureFaceError as exc:  # pragma: no cover - network/runtime errors
                _LOGGER.warning("Failed to prepare Azure Face person group: %s", exc)

    # ------------------------------------------------------------------
    def _create_client(self) -> tuple[Any | None, str | None]:
        if VisionFaceClient is not None and AzureKeyCredential is not None:
            try:
                client = VisionFaceClient(self.endpoint, AzureKeyCredential(self.api_key))  # type: ignore[arg-type]
                _LOGGER.info("Azure Face service initialised using azure-ai-vision-face")
                return client, "vision"
            except Exception as exc:  # pragma: no cover - runtime failure
                _LOGGER.error("Failed to initialise azure-ai-vision-face client: %s", exc)

        if CognitiveFaceClient is not None and CognitiveServicesCredentials is not None:
            try:
                client = CognitiveFaceClient(
                    self.endpoint, CognitiveServicesCredentials(self.api_key)
                )
                _LOGGER.info(
                    "Azure Face service initialised using azure-cognitiveservices-vision-face"
                )
                return client, "cognitive"
            except Exception as exc:  # pragma: no cover - runtime failure
                _LOGGER.error(
                    "Failed to initialise azure-cognitiveservices-vision-face client: %s", exc
                )

        if self.endpoint and self.api_key:
            _LOGGER.warning(
                "Azure Face SDK 未安裝，請安裝 azure-ai-vision-face 或 azure-cognitiveservices-vision-face"
            )
        return None, None

    # ------------------------------------------------------------------
    @property
    def can_describe_faces(self) -> bool:
        return self._client is not None

    @property
    def can_generate_ads(self) -> bool:
        # Azure Face 僅提供臉部偵測；我們保留一個簡單模板供廣告頁面使用。
        return True

    @property
    def can_manage_person_group(self) -> bool:
        return self.can_describe_faces and bool(self.person_group_id)

    # ------------------------------------------------------------------
    def analyze_face(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> FaceAnalysis:
        """Detect a face, return its description and any known Azure person mapping."""

        del mime_type

        if not self.can_describe_faces:
            raise AzureFaceError(
                "Azure Face service is not configured. Set AZURE_FACE_ENDPOINT and AZURE_FACE_KEY."
            )

        stream = io.BytesIO(image_bytes)
        try:
            faces = self._detect_with_stream(stream, return_face_id=True)
        except AzureFaceError:
            raise
        except Exception as exc:  # pragma: no cover - network/runtime errors
            raise AzureFaceError(f"Azure Face detection failed: {exc}") from exc

        if not faces:
            raise AzureFaceError("Azure Face did not detect any faces in the image")
        if len(faces) > 1:
            _LOGGER.debug("Azure Face detected %d faces, selecting the first", len(faces))

        face = faces[0]
        attributes = self._extract_face_attributes(face)
        description = self._format_face_description(attributes)
        face_id = self._extract_face_id(face)

        person_id = None
        person_name = None
        confidence = None
        if self.can_manage_person_group and face_id:
            try:
                self._ensure_person_group()
                results = self._identify_face_ids([face_id])
                candidate = self._select_best_candidate(results)
                if candidate is not None:
                    person_id = candidate["person_id"]
                    confidence = candidate.get("confidence")
                    person_name = self._resolve_person_name(person_id)
            except AzureFaceError as exc:
                _LOGGER.warning("Azure Face identify failed: %s", exc)

        if not description and not person_id:
            raise AzureFaceError("Azure Face returned no usable attributes for the detected face")

        analysis = FaceAnalysis(
            description=description or "",
            face_id=face_id,
            person_id=person_id,
            person_name=person_name,
            confidence=confidence,
        )
        _LOGGER.debug(
            "Azure Face analysis: id=%s, name=%s, confidence=%s, description=%s",
            analysis.person_id,
            analysis.person_name,
            analysis.confidence,
            analysis.description,
        )
        return analysis

    def describe_face(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Return a stable textual description for the supplied face image."""

        analysis = self.analyze_face(image_bytes, mime_type=mime_type)
        return analysis.description

    def register_person(
        self,
        member_id: str,
        image_bytes: bytes,
        *,
        user_data: str | None = None,
    ) -> str:
        """Add a new Azure Face person and associate the uploaded face image."""

        if not self.can_manage_person_group:
            raise AzureFaceError("Azure Face person group features are not available")

        self._ensure_person_group()
        person_client = self._get_person_group_person_client()
        stream = io.BytesIO(image_bytes)

        try:
            create_params = self._build_call_args(
                getattr(person_client, "create"),
                {
                    "person_group_id": self.person_group_id,
                    "name": member_id,
                    "user_data": user_data or member_id,
                },
            )
            person = person_client.create(**create_params)
        except Exception as exc:  # pragma: no cover - runtime failures
            raise AzureFaceError(f"Failed to create Azure Face person: {exc}") from exc

        person_id = self._extract_person_id(person)
        if not person_id:
            raise AzureFaceError("Azure Face did not return a person identifier")

        self._attach_face_to_person_stream(person_id, stream)

        self.train_person_group()

        return person_id

    # ------------------------------------------------------------------
    def add_face_to_person(self, person_id: str, image_bytes: bytes) -> None:
        """Associate an additional face image with an existing Azure person."""

        if not self.can_manage_person_group:
            raise AzureFaceError("Azure Face person group features are not available")

        self._ensure_person_group()
        stream = io.BytesIO(image_bytes)
        self._attach_face_to_person_stream(person_id, stream)

    def train_person_group(self, *, suppress_errors: bool = True) -> None:
        """Trigger training for the configured Azure Face person group."""

        if not self.can_manage_person_group:
            raise AzureFaceError("Azure Face person group features are not available")

        group_client = self._get_person_group_client()
        try:
            train_params = self._build_call_args(
                getattr(group_client, "train"),
                {"person_group_id": self.person_group_id},
            )
            group_client.train(**train_params)
        except Exception as exc:  # pragma: no cover - runtime failures
            message = f"Failed to train Azure Face person group: {exc}"
            if suppress_errors:
                _LOGGER.warning(message)
                return
            raise AzureFaceError(message) from exc

    def find_person_id_by_name(self, member_name: str) -> str | None:
        """Return the Azure person identifier registered with the given name."""

        if not self.can_manage_person_group:
            return None

        self._ensure_person_group()
        person_client = self._get_person_group_person_client()

        try:
            call_args = self._build_call_args(
                getattr(person_client, "list", None),
                {"person_group_id": self.person_group_id},
            )
            if call_args is None:
                raise AttributeError("list")
            persons = person_client.list(**call_args)
        except Exception as exc:  # pragma: no cover - runtime failures
            raise AzureFaceError(f"Failed to enumerate Azure Face persons: {exc}") from exc

        if persons is None:
            return None

        for person in persons:
            if isinstance(person, dict):
                name = person.get("name")
                user_data = person.get("user_data") or person.get("userData")
            else:
                name = getattr(person, "name", None)
                user_data = getattr(person, "user_data", None) or getattr(
                    person, "userData", None
                )

            if isinstance(name, str) and name == member_name:
                return self._extract_person_id(person)
            if isinstance(user_data, str) and user_data == member_name:
                return self._extract_person_id(person)

        return None

    # ------------------------------------------------------------------
    def _detect_with_stream(self, stream: io.BytesIO, *, return_face_id: bool = False):
        stream.seek(0)
        if self._client is None:
            raise AzureFaceError("Azure Face client is not available")

        face_operation = getattr(self._client, "face", None)
        kwargs = {
            "return_face_attributes": self._resolve_face_attributes(),
            "detection_model": self._detection_model,
            "recognition_model": self._recognition_model,
            "return_face_id": return_face_id,
        }

        if callable(face_operation):  # pragma: no cover - unexpected signature
            face_operation = face_operation()

        if face_operation is not None and hasattr(face_operation, "detect_with_stream"):
            detector = getattr(face_operation, "detect_with_stream")
        else:
            detector = getattr(self._client, "detect_with_stream", None)

        if detector is None:
            raise AzureFaceError("Azure Face client does not provide detect_with_stream")

        candidate_kwargs = {
            "image": stream,
            "stream": stream,
            "input": stream,
            "image_stream": stream,
            "detection_model": self._detection_model,
            "recognition_model": self._recognition_model,
            "return_face_attributes": kwargs["return_face_attributes"],
            "return_face_id": return_face_id,
        }
        call_args = self._build_call_args(detector, candidate_kwargs)
        try:
            return detector(**call_args)
        except TypeError:
            # Fallback to legacy signature where stream is the first positional argument.
            remaining_kwargs = {
                key: value
                for key, value in candidate_kwargs.items()
                if key not in {"image", "stream", "input", "image_stream"}
            }
            return detector(stream, **remaining_kwargs)

    def _resolve_face_attributes(self) -> Sequence[Any]:
        if FaceAttributeType is None:
            return self._DEFAULT_FACE_ATTRIBUTES
        return [
            FaceAttributeType.age,
            FaceAttributeType.gender,
            FaceAttributeType.glasses,
            FaceAttributeType.facial_hair,
            FaceAttributeType.emotion,
            FaceAttributeType.hair,
            FaceAttributeType.makeup,
            FaceAttributeType.accessories,
            FaceAttributeType.smile,
        ]

    @staticmethod
    def _extract_face_attributes(face: Any) -> Any:
        for attribute_name in ("face_attributes", "faceAttributes"):
            if hasattr(face, attribute_name):
                return getattr(face, attribute_name)
            if isinstance(face, dict) and attribute_name in face:
                return face[attribute_name]
        return face

    @staticmethod
    def _extract_face_id(face: Any) -> str | None:
        for attribute_name in ("face_id", "faceId", "id"):
            value = None
            if hasattr(face, attribute_name):
                value = getattr(face, attribute_name)
            elif isinstance(face, dict):
                value = face.get(attribute_name)
            if isinstance(value, str) and value:
                return value
        return None

    def _format_face_description(self, attrs: Any) -> str:
        parts: list[str] = []

        age = self._safe_get(attrs, "age")
        if isinstance(age, (int, float)) and age > 0:
            parts.append(f"約 {int(round(age)):d} 歲")

        gender = self._safe_get(attrs, "gender")
        if isinstance(gender, str) and gender:
            cleaned = gender.strip().lower()
            if cleaned in {"male", "female"}:
                parts.append("男性" if cleaned == "male" else "女性")
            else:
                parts.append(gender.strip())

        hair = self._describe_hair(attrs)
        if hair:
            parts.append(hair)

        glasses = self._describe_glasses(attrs)
        if glasses:
            parts.append(glasses)

        accessories = self._describe_accessories(attrs)
        if accessories:
            parts.append(accessories)

        facial_hair = self._describe_facial_hair(attrs)
        if facial_hair:
            parts.append(facial_hair)

        smile = self._safe_get(attrs, "smile")
        if isinstance(smile, (int, float)) and smile >= 0.5:
            parts.append("帶著笑容")

        emotion = self._describe_emotion(attrs)
        if emotion:
            parts.append(emotion)

        description = "、".join(part for part in parts if part)
        return description.strip()

    @staticmethod
    def _safe_get(obj: Any, attribute: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(attribute)
        return getattr(obj, attribute, None)

    def _describe_hair(self, attrs: Any) -> str:
        hair = self._safe_get(attrs, "hair")
        if hair is None:
            return ""

        length = self._safe_get(hair, "hair_length") or self._safe_get(hair, "length")
        if isinstance(length, str):
            length = length.replace("_", " ")

        colours = self._safe_get(hair, "hair_color") or self._safe_get(hair, "hairColor")
        colour_label = ""
        best_confidence = 0.0
        if colours:
            for colour in colours:
                if isinstance(colour, dict):
                    label = colour.get("color")
                    confidence = colour.get("confidence", 0.0)
                else:
                    label = getattr(colour, "color", None)
                    confidence = getattr(colour, "confidence", 0.0)
                try:
                    score = float(confidence)
                except (TypeError, ValueError):
                    score = 0.0
                if score >= best_confidence and isinstance(label, str) and label:
                    best_confidence = score
                    colour_label = label.replace("_", " ")

        segments = []
        if colour_label:
            segments.append(colour_label)
        if isinstance(length, str) and length:
            segments.append(length)
        if segments:
            return "".join(segments) + "髮"
        return ""

    def _describe_glasses(self, attrs: Any) -> str:
        glasses = self._safe_get(attrs, "glasses")
        if isinstance(glasses, str):
            cleaned = glasses.strip().lower()
            if cleaned and cleaned not in {"noglasses", "no_glasses", "none"}:
                if "sunglasses" in cleaned:
                    return "配戴墨鏡"
                return "配戴眼鏡"
        return ""

    def _describe_accessories(self, attrs: Any) -> str:
        accessories = self._safe_get(attrs, "accessories")
        if not accessories:
            return ""
        labels: list[str] = []
        for accessory in accessories:
            if isinstance(accessory, dict):
                label = accessory.get("type") or accessory.get("type_")
                confidence = accessory.get("confidence")
            else:
                label = getattr(accessory, "type", None) or getattr(accessory, "type_", None)
                confidence = getattr(accessory, "confidence", None)
            try:
                score = float(confidence) if confidence is not None else 0.0
            except (TypeError, ValueError):
                score = 0.0
            if score < 0.3 or not isinstance(label, str):
                continue
            cleaned = label.split(".")[-1].replace("_", " ")
            labels.append(cleaned)
        if labels:
            return f"配戴{'、'.join(labels)}"
        return ""

    def _describe_facial_hair(self, attrs: Any) -> str:
        facial_hair = self._safe_get(attrs, "facial_hair") or self._safe_get(attrs, "facialHair")
        if not facial_hair:
            return ""
        scores = []
        for key in ("beard", "moustache", "sideburns"):
            value = self._safe_get(facial_hair, key)
            if isinstance(value, (int, float)):
                scores.append(float(value))
        if any(score > 0.5 for score in scores):
            return "留有鬍鬚"
        return ""

    def _describe_emotion(self, attrs: Any) -> str:
        emotion = self._safe_get(attrs, "emotion")
        if not emotion:
            return ""

        mapping = {
            "happiness": "開朗",
            "sadness": "憂鬱",
            "neutral": "神情平和",
            "anger": "表情嚴肅",
            "surprise": "略顯驚訝",
        }
        best_label = ""
        best_score = 0.0
        for name, value in self._iter_properties(emotion):
            if name in {"additional_properties"}:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score > best_score:
                best_score = score
                best_label = mapping.get(name, "")
        if best_label and best_score >= 0.5:
            return best_label
        return ""

    @staticmethod
    def _iter_properties(obj: Any) -> Iterator[tuple[str, Any]]:
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield key, value
        else:
            for name in dir(obj):
                if name.startswith("_"):
                    continue
                value = getattr(obj, name, None)
                if callable(value):
                    continue
                yield name, value

    # ------------------------------------------------------------------
    def _identify_face_ids(self, face_ids: Sequence[str]):
        if not face_ids:
            return []
        if not self.can_manage_person_group:
            raise AzureFaceError("Azure Face person group features are not available")

        face_operation = getattr(self._client, "face", None)
        if callable(face_operation):  # pragma: no cover - unexpected signature
            face_operation = face_operation()

        if face_operation is not None and hasattr(face_operation, "identify"):
            identify = getattr(face_operation, "identify")
        else:
            identify = getattr(self._client, "identify", None)

        if identify is None:
            adapter = self._get_rest_adapter()
            if adapter is not None:
                identify = adapter.identify

        if identify is None:
            raise AzureFaceError("Azure Face client does not provide identify")

        call_args = self._build_call_args(
            identify,
            {
                "person_group_id": self.person_group_id,
                "face_ids": list(face_ids),
                "max_num_of_candidates_return": 1,
                "max_num_of_candidates": 1,
            },
        )

        try:
            return identify(**call_args)
        except Exception as exc:  # pragma: no cover - runtime failures
            raise AzureFaceError(f"Azure Face identification failed: {exc}") from exc

    def _select_best_candidate(self, results: Any) -> dict[str, Any] | None:
        best_candidate: dict[str, Any] | None = None
        best_confidence = 0.0

        if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
            iterable = results
        else:
            iterable = [results]
        for result in iterable:
            candidates = getattr(result, "candidates", None) or (
                result.get("candidates") if isinstance(result, dict) else None
            )
            if not candidates:
                continue
            for candidate in candidates:
                person_id = getattr(candidate, "person_id", None) or (
                    candidate.get("person_id") if isinstance(candidate, dict) else None
                )
                if person_id is None:
                    person_id = getattr(candidate, "personId", None)
                confidence = getattr(candidate, "confidence", None)
                if isinstance(candidate, dict):
                    confidence = candidate.get("confidence", confidence)
                try:
                    score = float(confidence) if confidence is not None else 0.0
                except (TypeError, ValueError):
                    score = 0.0
                if not isinstance(person_id, str) or not person_id:
                    continue
                if best_candidate is None or score > best_confidence:
                    best_confidence = score
                    best_candidate = {
                        "person_id": person_id,
                        "confidence": score,
                    }
        return best_candidate

    def _get_rest_adapter(self) -> _AzureFaceRestAdapter | None:
        if self._rest_adapter is not None:
            return self._rest_adapter
        if not self.endpoint or not self.api_key:
            return None
        try:
            self._rest_adapter = _AzureFaceRestAdapter(
                self.endpoint,
                self.api_key,
                detection_model=self._detection_model,
                recognition_model=self._recognition_model,
            )
        except AzureFaceError as exc:  # pragma: no cover - runtime errors
            _LOGGER.warning("Failed to initialise Azure Face REST fallback: %s", exc)
            return None
        return self._rest_adapter

    def _resolve_person_name(self, person_id: str | None) -> str | None:
        if not person_id or not self.can_manage_person_group:
            return None

        person_client = self._get_person_group_person_client()
        try:
            call_args = self._build_call_args(
                getattr(person_client, "get"),
                {
                    "person_group_id": self.person_group_id,
                    "person_id": person_id,
                },
            )
            person = person_client.get(**call_args)
        except Exception as exc:  # pragma: no cover - runtime failures
            _LOGGER.warning("Failed to fetch Azure Face person metadata: %s", exc)
            return None

        for attribute in ("name", "user_data", "userData"):
            value = getattr(person, attribute, None)
            if isinstance(person, dict):
                value = value or person.get(attribute)
            if isinstance(value, str) and value:
                return value
        return None

    def _ensure_person_group(self) -> None:
        if not self.can_manage_person_group:
            raise AzureFaceError("Azure Face person group features are not available")
        if self._person_group_ready:
            return
        try:
            group_client = self._get_person_group_client()
            call_args = self._build_call_args(
                getattr(group_client, "get"),
                {"person_group_id": self.person_group_id},
            )
            group_client.get(**call_args)
        except Exception:
            try:
                self._create_person_group()
            except AzureFaceError as exc:
                raise exc
        else:
            self._person_group_ready = True

    def _create_person_group(self) -> None:
        group_client = self._get_person_group_client()
        try:
            call_args = self._build_call_args(
                getattr(group_client, "create"),
                {
                    "person_group_id": self.person_group_id,
                    "name": self.person_group_id,
                    "recognition_model": self._recognition_model,
                },
            )
            group_client.create(**call_args)
            self._person_group_ready = True
        except Exception as exc:  # pragma: no cover - runtime failures
            raise AzureFaceError(f"Failed to create Azure Face person group: {exc}") from exc

    def _get_person_group_client(self):
        candidate_attributes = (
            "person_group",
            "personGroup",
            "person_group_client",
            "personGroupClient",
            "person_group_operations",
            "personGroupOperations",
            "get_person_group_client",
            "get_person_group",
        )

        last_error: Exception | None = None
        for attribute in candidate_attributes:
            group_client = getattr(self._client, attribute, None)
            if group_client is None:
                continue

            if callable(group_client):
                try:
                    call_args = self._build_call_args(
                        group_client,
                        {"person_group_id": self.person_group_id},
                    )
                    group_client = group_client(**call_args)
                except TypeError:  # pragma: no cover - signature without kwargs
                    try:
                        group_client = group_client()
                    except Exception as exc:  # pragma: no cover - runtime failure
                        last_error = exc
                        continue
                except Exception as exc:  # pragma: no cover - runtime failure
                    last_error = exc
                    continue

            if group_client is not None:
                return group_client

        adapter = self._get_rest_adapter()
        if adapter is not None:
            _LOGGER.info("Using Azure Face REST fallback for person group operations")
            return adapter.person_group

        if last_error is not None:
            raise AzureFaceError(
                f"Failed to obtain Azure Face person group client: {last_error}"
            ) from last_error
        raise AzureFaceError("Azure Face client does not expose person group operations")

    def _get_person_group_person_client(self):
        candidate_attributes = (
            "person_group_person",
            "personGroupPerson",
            "person_group_person_client",
            "personGroupPersonClient",
            "person_group_person_operations",
            "personGroupPersonOperations",
            "get_person_group_person_client",
            "get_person_group_person",
        )

        last_error: Exception | None = None
        for attribute in candidate_attributes:
            person_client = getattr(self._client, attribute, None)
            if person_client is None:
                continue

            if callable(person_client):
                try:
                    call_args = self._build_call_args(
                        person_client,
                        {"person_group_id": self.person_group_id},
                    )
                    person_client = person_client(**call_args)
                except TypeError:  # pragma: no cover - signature without kwargs
                    try:
                        person_client = person_client()
                    except Exception as exc:  # pragma: no cover - runtime failure
                        last_error = exc
                        continue
                except Exception as exc:  # pragma: no cover - runtime failure
                    last_error = exc
                    continue

            if person_client is not None:
                return person_client

        adapter = self._get_rest_adapter()
        if adapter is not None:
            _LOGGER.info("Using Azure Face REST fallback for person operations")
            return adapter.person_group_person

        if last_error is not None:
            raise AzureFaceError(
                f"Failed to obtain Azure Face person client: {last_error}"
            ) from last_error
        raise AzureFaceError("Azure Face client does not expose person operations")

    @staticmethod
    def _extract_person_id(person: Any) -> str | None:
        if isinstance(person, dict):
            for key in ("person_id", "personId", "id"):
                value = person.get(key)
                if isinstance(value, str) and value:
                    return value
        else:
            for attribute in ("person_id", "personId", "id"):
                value = getattr(person, attribute, None)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _build_call_args(func: Any, candidate_kwargs: dict[str, Any], *, allow_positional: bool = False):
        if func is None:
            return None
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            return candidate_kwargs

        parameters = signature.parameters
        call_args: dict[str, Any] = {}
        for name, value in candidate_kwargs.items():
            if name in parameters:
                call_args[name] = value
        if allow_positional and any(param.kind == inspect.Parameter.POSITIONAL_ONLY for param in parameters.values()):
            # Legacy SDK expects the image stream as the first positional argument; leave kwargs empty so
            # the caller can retry with positional usage.
            call_args.update({k: v for k, v in candidate_kwargs.items() if k in parameters})
            call_args.update({k: v for k, v in candidate_kwargs.items() if k not in call_args})
            return call_args

        # Fall back to any parameters that look compatible when names do not match exactly.
        for name, param in parameters.items():
            if name in call_args:
                continue
            lowered = name.lower()
            for candidate, value in candidate_kwargs.items():
                if candidate.lower() == lowered:
                    call_args[name] = value
        return call_args

    # ------------------------------------------------------------------
    def _attach_face_to_person_stream(self, person_id: str, stream: io.BytesIO) -> None:
        person_client = self._get_person_group_person_client()
        stream.seek(0)
        try:
            add_face_params = self._build_call_args(
                getattr(person_client, "add_face_from_stream", None),
                {
                    "person_group_id": self.person_group_id,
                    "person_id": person_id,
                    "image": stream,
                    "detection_model": self._detection_model,
                },
            )
            if add_face_params is None:
                raise AttributeError("add_face_from_stream")
            person_client.add_face_from_stream(**add_face_params)
        except Exception as exc:  # pragma: no cover - runtime failures
            raise AzureFaceError(f"Failed to attach face to Azure person: {exc}") from exc

    # ------------------------------------------------------------------
    def generate_ad_copy(self, member_id: str, purchases: Iterable[dict[str, object]]) -> AdCreative:
        """Produce a simple, deterministic advertising copy template."""

        purchase_list = list(purchases)
        if purchase_list:
            latest = purchase_list[0]
            item = str(latest.get("item", "精選商品"))
            discount = latest.get("discount")
            if isinstance(discount, (int, float)) and discount > 0:
                discount_text = f"{int(discount * 100)}% OFF"
            else:
                discount_text = "限時加購禮"
            headline = f"會員 {member_id}，歡迎回來！"
            subheading = f"上次購買：{item}"
            highlight = f"今天享有 {discount_text}，立即入手！"
        else:
            headline = f"歡迎加入，會員 {member_id}!"
            subheading = "首次來店禮已為您準備"
            highlight = "人氣咖啡豆搭配手工點心，今日限定好禮"
        return AdCreative(headline=headline, subheading=subheading, highlight=highlight)


class _AzureFaceRestAdapter:
    """Fallback helper that issues REST requests when SDK operations are missing."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        detection_model: str,
        recognition_model: str,
    ) -> None:
        self._api = _AzureFaceRestAPI(endpoint, api_key)
        self.person_group = _AzureFaceRestPersonGroupOperations(
            self._api, recognition_model=recognition_model
        )
        self.person_group_person = _AzureFaceRestPersonOperations(
            self._api, detection_model=detection_model
        )

    def identify(
        self,
        *,
        person_group_id: str,
        face_ids: Sequence[str],
        max_num_of_candidates_return: int | None = None,
        max_num_of_candidates: int | None = None,
    ) -> Any:
        max_candidates = max_num_of_candidates_return or max_num_of_candidates or 1
        try:
            value = int(max_candidates)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            value = 1
        if value <= 0:
            value = 1
        return self._api.identify(
            person_group_id=person_group_id,
            face_ids=list(face_ids),
            max_candidates=value,
        )


class _AzureFaceRestPersonGroupOperations:
    """REST implementation of the person group operations used by the service."""

    def __init__(self, api: "_AzureFaceRestAPI", *, recognition_model: str) -> None:
        self._api = api
        self._recognition_model = recognition_model

    def get(self, *, person_group_id: str) -> Any:
        return self._api.get_person_group(person_group_id)

    def create(
        self,
        *,
        person_group_id: str,
        name: str,
        recognition_model: str | None = None,
    ) -> Any:
        return self._api.create_person_group(
            person_group_id,
            name=name,
            recognition_model=recognition_model or self._recognition_model,
        )

    def train(self, *, person_group_id: str) -> Any:
        return self._api.train_person_group(person_group_id)


class _AzureFaceRestPersonOperations:
    """REST implementation of the person operations used by the service."""

    def __init__(self, api: "_AzureFaceRestAPI", *, detection_model: str) -> None:
        self._api = api
        self._detection_model = detection_model

    def create(
        self,
        *,
        person_group_id: str,
        name: str,
        user_data: str | None = None,
    ) -> Any:
        return self._api.create_person(
            person_group_id,
            name=name,
            user_data=user_data,
        )

    def add_face_from_stream(
        self,
        *,
        person_group_id: str,
        person_id: str,
        image: Any,
        detection_model: str | None = None,
    ) -> Any:
        return self._api.add_person_face(
            person_group_id,
            person_id,
            image,
            detection_model=detection_model or self._detection_model,
        )

    def list(self, *, person_group_id: str) -> Any:
        return self._api.list_persons(person_group_id)

    def get(self, *, person_group_id: str, person_id: str) -> Any:
        return self._api.get_person(person_group_id, person_id)


class _AzureFaceRestAPI:
    """Minimal REST client for Azure Face person group operations."""

    _TIMEOUT_SECONDS = 15

    def __init__(self, endpoint: str, api_key: str) -> None:
        if not endpoint or not api_key:
            raise AzureFaceError("Azure Face endpoint and key are required for REST fallback")
        base = endpoint.rstrip("/")
        lower = base.lower()
        if "/face/v1.0" in lower:
            self._base_url = base
        else:
            self._base_url = f"{base}/face/v1.0"
        self._api_key = api_key

    # ------------------------------------------------------------------
    def identify(self, *, person_group_id: str, face_ids: Sequence[str], max_candidates: int) -> Any:
        payload = {
            "personGroupId": person_group_id,
            "faceIds": list(face_ids),
            "maxNumOfCandidatesReturned": max(1, int(max_candidates)),
        }
        return self._request("POST", "identify", json_body=payload, expected_status=(200,))

    def get_person_group(self, person_group_id: str) -> Any:
        return self._request("GET", f"persongroups/{person_group_id}", expected_status=(200,))

    def create_person_group(
        self,
        person_group_id: str,
        *,
        name: str,
        recognition_model: str | None,
    ) -> Any:
        payload = {"name": name}
        if recognition_model:
            payload["recognitionModel"] = recognition_model
        return self._request(
            "PUT",
            f"persongroups/{person_group_id}",
            json_body=payload,
            expected_status=(200, 202, 204),
        )

    def train_person_group(self, person_group_id: str) -> Any:
        return self._request(
            "POST",
            f"persongroups/{person_group_id}/train",
            expected_status=(202, 200, 204),
        )

    def list_persons(self, person_group_id: str) -> Any:
        result = self._request(
            "GET",
            f"persongroups/{person_group_id}/persons",
            expected_status=(200,),
        )
        return result or []

    def get_person(self, person_group_id: str, person_id: str) -> Any:
        return self._request(
            "GET",
            f"persongroups/{person_group_id}/persons/{person_id}",
            expected_status=(200,),
        )

    def create_person(
        self,
        person_group_id: str,
        *,
        name: str,
        user_data: str | None,
    ) -> Any:
        payload = {"name": name}
        if user_data:
            payload["userData"] = user_data
        return self._request(
            "POST",
            f"persongroups/{person_group_id}/persons",
            json_body=payload,
            expected_status=(200, 201),
        )

    def add_person_face(
        self,
        person_group_id: str,
        person_id: str,
        image: Any,
        *,
        detection_model: str | None,
    ) -> Any:
        params = {}
        if detection_model:
            params["detectionModel"] = detection_model
        return self._request(
            "POST",
            f"persongroups/{person_group_id}/persons/{person_id}/persistedFaces",
            params=params,
            data=image,
            headers={"Content-Type": "application/octet-stream"},
            expected_status=(200, 202),
        )

    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any | None = None,
        data: Any | None = None,
        headers: dict[str, str] | None = None,
        expected_status: Sequence[int] = (200,),
    ) -> Any:
        url = self._build_url(path, params)
        request_headers = {"Ocp-Apim-Subscription-Key": self._api_key}
        if headers:
            request_headers.update(headers)

        body: bytes | None = None
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        elif data is not None:
            if hasattr(data, "read"):
                body = data.read()
            else:
                body = data
            if body is None:
                body = b""
            if not isinstance(body, (bytes, bytearray)):
                body = bytes(body)
            request_headers.setdefault("Content-Type", "application/octet-stream")

        req = urllib_request.Request(url, data=body, headers=request_headers, method=method.upper())
        try:
            with urllib_request.urlopen(req, timeout=self._TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", response.getcode())
                payload = response.read()
        except urllib_error.HTTPError as exc:  # pragma: no cover - network/runtime errors
            status = exc.code
            try:
                detail = exc.read()
            except Exception:  # pragma: no cover - read failure
                detail = b""
            message = detail.decode("utf-8", "ignore").strip() or str(exc)
            raise AzureFaceError(f"Azure Face REST request failed: {status} {message}") from exc
        except Exception as exc:  # pragma: no cover - network/runtime errors
            raise AzureFaceError(f"Azure Face REST request failed: {exc}") from exc

        if status not in expected_status:
            raise AzureFaceError(f"Azure Face REST request returned status {status}")
        if not payload:
            return None
        text = payload.decode("utf-8", "ignore").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return payload

    def _build_url(self, path: str, params: dict[str, Any] | None) -> str:
        clean_path = path.lstrip("/")
        if clean_path:
            url = f"{self._base_url}/{clean_path}"
        else:
            url = self._base_url
        if params:
            filtered = {key: value for key, value in params.items() if value is not None}
            if filtered:
                url = f"{url}?{urllib_parse.urlencode(filtered)}"
        return url

__all__ = ["AdCreative", "AzureFaceError", "AzureFaceService", "FaceAnalysis"]
