"""Route blueprints for the ESP32-CAM backend."""

from .adgen import adgen_blueprint, generate_vertex_ad

__all__ = ["adgen_blueprint", "generate_vertex_ad"]
