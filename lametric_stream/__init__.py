"""
LaMetric Stream Library
======================

A Python library for streaming custom content to LaMetric TIME devices using the LMSP protocol.

This library allows you to:
- Display text with custom fonts and character sets
- Show solid colors and gradients
- Create custom animations
- Stream real-time data visualizations
- Create independent screen regions with separate content
"""

__version__ = "0.2.0"

# Export public API
from .client import LMStream, ContentEncoding, RenderMode, FillType, PostProcessType
from .exceptions import LMStreamError, APIError, ProtocolError
from .utils import (
    CanvasArea,
    ScreenRegion,
    RegionManager,
    RegionContent,
    TextContent,
    ColorContent,
    AnimationContent,
)

__all__ = [
    "LMStream",
    "ContentEncoding",
    "RenderMode",
    "FillType",
    "PostProcessType",
    "LMStreamError",
    "APIError",
    "ProtocolError",
    "CanvasArea",
    "ScreenRegion",
    "RegionManager",
    "RegionContent",
    "TextContent",
    "ColorContent",
    "AnimationContent",
]
