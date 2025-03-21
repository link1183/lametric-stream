"""
Utility functions for the LaMetric Stream library.
"""

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CanvasArea:
    """Represents a canvas area for streaming."""

    width: int
    height: int
    pixels: List[Tuple[int, int, int]]

    x: int = 0
    y: int = 0

    def validate(self) -> None:
        """Validate canvas area dimensions and pixel data."""
        if not (0 <= self.x and 0 <= self.y):
            raise ValueError(f"Invalid coordinates: ({self.x}, {self.y})")

        if not (0 < self.width and 0 < self.height):
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")

        expected_pixels = self.width * self.height
        if len(self.pixels) != expected_pixels:
            raise ValueError(
                f"Expected {expected_pixels} pixels, got {len(self.pixels)}"
            )

        for i, pixel in enumerate(self.pixels):
            if len(pixel) != 3:
                raise ValueError(f"Pixel {i} has {len(pixel)} values, expected 3 (RGB)")

            for j, value in enumerate(pixel):
                if not (0 <= value <= 255):
                    raise ValueError(
                        f"Pixel {i}, component {j} has value {value}, expected 0-255"
                    )


class RequestsRetrySession:
    """Creates a requests session with retry capabilities."""

    @staticmethod
    def create(
        retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: Tuple[int, int, int, int] = (500, 502, 503, 504),
    ) -> requests.Session:
        """Create a requests session with retry configuration.

        Args:
            retries: Number of retry attempts
            backoff_factor: Backoff factor between retries
            status_forcelist: Status codes to retry on

        Returns:
            Configured requests.Session object
        """
        session = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
