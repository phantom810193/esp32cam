"""Advanced face processing pipeline using Real-ESRGAN, GFPGAN, and InsightFace."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

_LOGGER = logging.getLogger(__name__)


class AdvancedFacePipelineError(RuntimeError):
    """Raised when the advanced face pipeline cannot produce an embedding."""


@dataclass
class ProcessedFace:
    """Container for the output of :class:`AdvancedFacePipeline`."""

    embedding: np.ndarray
    image: np.ndarray
    bbox: Sequence[int] | None = None
    quality: float | None = None


class AdvancedFacePipeline:
    """Optional enhancement pipeline that combines multiple open source models.

    The pipeline performs three high level steps:

    1. **Real-ESRGAN** upscales and denoises the incoming frame to recover detail.
    2. **GFPGAN** restores facial regions using a generative prior.
    3. **InsightFace** (ArcFace model) extracts a 512-dimension embedding.

    All of the dependencies are optional.  When any component is unavailable the
    pipeline gracefully skips that stage instead of raising on import.  Only the
    InsightFace embedding step is mandatory for the pipeline to be considered
    available.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        providers: Iterable[str] | None = None,
        insightface_model: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
    ) -> None:
        self.model_dir = Path(model_dir or Path(__file__).resolve().parent / "models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.providers = list(providers or ["CPUExecutionProvider"])
        self.insightface_model = insightface_model
        self.det_size = det_size

        self._upscaler = None
        self._restorer = None
        self._face_app = None
        self._last_error: str | None = None

        self._initialise_components()

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._face_app is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ------------------------------------------------------------------
    def _initialise_components(self) -> None:
        """Attempt to import and configure all optional dependencies."""

        self._last_error = None
        face_app = self._initialise_insightface()
        if face_app is None:
            self._last_error = (
                "InsightFace dependencies are missing – install insightface and "
                "onnxruntime-gpu/onnxruntime to enable the advanced pipeline."
            )
            return

        self._face_app = face_app
        self._upscaler = self._initialise_realesrgan()
        self._restorer = self._initialise_gfpgan(self._upscaler)

    def _initialise_insightface(self):
        try:
            insightface_app = importlib.import_module("insightface.app")
        except ImportError as exc:  # pragma: no cover - optional dependency
            _LOGGER.info("InsightFace not available: %s", exc)
            return None

        face_analysis = getattr(insightface_app, "FaceAnalysis", None)
        if face_analysis is None:  # pragma: no cover - defensive
            _LOGGER.warning("insightface.app.FaceAnalysis not found")
            return None

        try:
            app = face_analysis(name=self.insightface_model, providers=self.providers)
            app.prepare(ctx_id=0, det_size=self.det_size)
            _LOGGER.info(
                "InsightFace initialised with model '%s' (providers=%s)",
                self.insightface_model,
                ",".join(self.providers),
            )
            return app
        except Exception as exc:  # pragma: no cover - runtime setup failure
            _LOGGER.warning("Failed to initialise InsightFace: %s", exc)
            return None

    def _initialise_realesrgan(self):
        try:
            realesrgan = importlib.import_module("realesrgan")
            basicsr_arch = importlib.import_module("basicsr.archs.rrdbnet_arch")
        except ImportError as exc:  # pragma: no cover - optional dependency
            _LOGGER.info("Real-ESRGAN not available: %s", exc)
            return None

        model_path = self.model_dir / "RealESRGAN_x4plus.pth"
        if not model_path.exists():
            _LOGGER.info("Real-ESRGAN weights not found at %s", model_path)
            return None

        rrdbnet_cls = getattr(basicsr_arch, "RRDBNet", None)
        realesrganer_cls = getattr(realesrgan, "RealESRGANer", None)
        if rrdbnet_cls is None or realesrganer_cls is None:  # pragma: no cover
            _LOGGER.warning("Real-ESRGAN components missing from the installed package")
            return None

        try:
            model = rrdbnet_cls(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            upscaler = realesrganer_cls(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
            )
            _LOGGER.info("Real-ESRGAN initialised with weights at %s", model_path)
            return upscaler
        except Exception as exc:  # pragma: no cover - runtime setup failure
            _LOGGER.warning("Failed to initialise Real-ESRGAN: %s", exc)
            return None

    def _initialise_gfpgan(self, upscaler):
        try:
            gfpgan = importlib.import_module("gfpgan")
        except ImportError as exc:  # pragma: no cover - optional dependency
            _LOGGER.info("GFPGAN not available: %s", exc)
            return None

        gfpgan_cls = getattr(gfpgan, "GFPGANer", None)
        if gfpgan_cls is None:  # pragma: no cover - defensive
            _LOGGER.warning("gfpgan.GFPGANer not found in installed package")
            return None

        model_path = self.model_dir / "GFPGANv1.4.pth"
        if not model_path.exists():
            _LOGGER.info("GFPGAN weights not found at %s", model_path)
            return None

        try:
            restorer = gfpgan_cls(
                model_path=str(model_path),
                upscale=4,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upscaler,
            )
            _LOGGER.info("GFPGAN initialised with weights at %s", model_path)
            return restorer
        except Exception as exc:  # pragma: no cover - runtime setup failure
            _LOGGER.warning("Failed to initialise GFPGAN: %s", exc)
            return None

    # ------------------------------------------------------------------
    def process(self, image_bytes: bytes, mime_type: str | None = None) -> ProcessedFace:
        if not self.is_available:
            raise AdvancedFacePipelineError(
                self._last_error or "InsightFace pipeline is not available"
            )

        try:
            image = Image.open(BytesIO(image_bytes))
        except Exception as exc:
            raise AdvancedFacePipelineError(f"Unable to decode image: {exc}") from exc

        image = image.convert("RGB")
        frame = np.array(image)
        if frame.ndim != 3:
            raise AdvancedFacePipelineError("Expected a colour image with three channels")

        enhanced = self._apply_enhancers(frame)
        faces = self._extract_faces(enhanced)
        if not faces:
            raise AdvancedFacePipelineError("No face detected by InsightFace")

        faces = sorted(
            faces,
            key=lambda face: float(getattr(face, "det_score", 0.0)),
            reverse=True,
        )
        best = faces[0]
        embedding = getattr(best, "normed_embedding", None) or getattr(best, "embedding", None)
        if embedding is None:
            raise AdvancedFacePipelineError("InsightFace did not return an embedding")

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise AdvancedFacePipelineError("InsightFace embedding has unexpected shape")

        bbox = getattr(best, "bbox", None)
        if bbox is not None:
            try:
                bbox = tuple(int(x) for x in np.asarray(bbox).tolist())
            except Exception:  # pragma: no cover - defensive
                bbox = None

        quality = getattr(best, "det_score", None)
        if quality is not None:
            try:
                quality = float(quality)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                quality = None

        return ProcessedFace(embedding=vector, image=enhanced, bbox=bbox, quality=quality)

    # ------------------------------------------------------------------
    def _apply_enhancers(self, frame: np.ndarray) -> np.ndarray:
        working = frame
        if self._upscaler is not None:
            try:
                working, _ = self._upscaler.enhance(working, outscale=4)
            except Exception as exc:  # pragma: no cover - runtime failure
                _LOGGER.warning("Real-ESRGAN enhancement failed: %s", exc)
                working = frame

        if self._restorer is not None:
            try:
                _c, _s, restored = self._restorer.enhance(
                    working,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
                if restored is not None:
                    working = restored
            except Exception as exc:  # pragma: no cover - runtime failure
                _LOGGER.warning("GFPGAN restoration failed: %s", exc)

        return working

    def _extract_faces(self, frame: np.ndarray) -> Sequence[object]:
        try:
            return self._face_app.get(frame)
        except Exception as exc:
            raise AdvancedFacePipelineError(f"InsightFace inference failed: {exc}") from exc

