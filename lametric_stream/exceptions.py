"""
Exception classes for the LaMetric Stream library.
"""


class LMStreamError(Exception):
    """Base exception for LMStream errors."""

    pass


class APIError(LMStreamError):
    """Raised when API requests fail."""

    pass


class ProtocolError(LMStreamError):
    """Raised when there's an issue with the LMSP protocol."""

    pass
