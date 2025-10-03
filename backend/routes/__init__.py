"""Route blueprints for the ESP32-CAM backend."""

from .adgen import adgen_blueprint
from .ads import ads_blueprint

__all__ = ["adgen_blueprint", "ads_blueprint"]
