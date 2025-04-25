"""
Main LMStream client for the LaMetric Stream library.
"""

import socket
import struct
import time
import base64
import binascii
import logging
import threading
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union, Any

import requests
from requests.models import ProtocolError

from lametric_stream.exceptions import APIError, LMStreamError

from .utils import (
    CanvasArea,
    RequestsRetrySession,
    RegionManager,
    ScreenRegion,
)
from .fonts import FONT_5X7, FONT_DESCENDERS


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lametric_stream")


class ContentEncoding(Enum):
    """Content encoding types supported by LMSP protocol."""

    RAW = 0x00
    PNG = 0x01
    JPEG = 0x02
    GIF = 0x03


class RenderMode(Enum):
    """Render modes supported by LaMetric devices."""

    PIXEL = "pixel"
    TRIANGLE = "triangle"


class FillType(Enum):
    """Fill types for canvas rendering."""

    SCALE = "scale"
    TILE = "tile"


class PostProcessType(Enum):
    """Post-processing effect types."""

    NONE = "none"
    EFFECT = "effect"


class LMStream:
    """Client for streaming content to LaMetric devices using LMSP protocol."""

    def __init__(
        self, api_key: str, ip: str, max_retries: int = 3, update_interval: float = 0.05
    ):
        """Initialize the LaMetric Streaming client.

        Args:
            ip: IP address of the LaMetric device
            api_key: API key for the device
            max_retries: Maximum number of retries for API requests
            update_interval: Time in seconds between region updates
        """

        self.ip = ip
        self.api_key = api_key

        self.base_url = f"http://{self.ip}:8080/api/v2/device"
        self.stream_url = f"{self.base_url}/stream"
        self.headers = self._create_auth_header()

        self.session = RequestsRetrySession.create(retries=max_retries)

        self.port = None
        self.session_id = None
        self.canvas_size = {}
        self.is_streaming = False
        self._socket = None

        # Region management
        self.region_manager = None
        self.update_interval = update_interval
        self._update_thread = None
        self._update_running = False

    def __enter__(self):
        """Context manager entry point - starts streaming session."""
        self._get_device_info()

        self.port = self.get_streaming_port()
        self.session_id = self.start_streaming_session()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)

        # Initialize region manager with canvas size
        self.region_manager = RegionManager(
            canvas_width=self.canvas_size.get("width", 0),
            canvas_height=self.canvas_size.get("height", 0),
        )
        self.region_manager.set_update_callback(self._on_regions_updated)

        # Start update thread
        self._start_update_thread()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit point - stops streaming and cleans up."""
        try:
            # Stop update thread
            self._stop_update_thread()

            if self.is_streaming:
                self.stop_streaming()
        finally:
            if self._socket:
                self._socket.close()
                self._socket = None

    def _create_auth_header(self) -> Dict[str, str]:
        """Create the authentication header for API requests."""
        auth_str = f"dev:{self.api_key}"
        auth_base64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        return {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/json",
        }

    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information including canvas size.

        Returns:
            Dict containing device information

        Raises:
            APIError: If the API request fails
        """
        try:
            response = self.session.get(self.stream_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            canvas_info = data.get("canvas", {})
            pixel_info = canvas_info.get("pixel", {}).get("size", {})

            self.canvas_size = {
                "width": pixel_info.get("width", 0),
                "height": pixel_info.get("height", 0),
            }

            logger.info(
                f"Device info retrieved. Canvas size: {self.canvas_size['width']}x{self.canvas_size['height']}"
            )
            return data

        except requests.RequestException as e:
            raise APIError(f"Failed to get device info: {str(e)}") from e

    def get_streaming_port(self) -> int:
        """Get the UDP port for streaming.

        Returns:
            Port number to use for streaming

        Raises:
            APIError: If the API request fails
        """
        try:
            response = self.session.get(self.stream_url, headers=self.headers)
            response.raise_for_status()
            port = response.json().get("port")

            if not port:
                raise APIError("No port received from device")

            logger.info(f"Streaming port: {port}")
            return port

        except requests.RequestException as e:
            raise APIError(f"Failed to get streaming port: {str(e)}") from e

    def start_streaming_session(
        self,
        fill_type: Union[FillType, str] = FillType.SCALE,
        render_mode: Union[RenderMode, str] = RenderMode.PIXEL,
        post_process_type: Union[PostProcessType, str] = PostProcessType.NONE,
        effect_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a streaming session with the device.

        Args:
            fill_type: How to fill the canvas ("scale" or "tile")
            render_mode: Rendering mode ("pixel" or "triangle")
            post_process_type: Post-processing type ("none" or "effect")
            effect_params: Parameters for the post-processing effect

        Returns:
            Session ID for the streaming session

        Raises:
            APIError: If the API request fails
        """
        if isinstance(fill_type, FillType):
            fill_type = fill_type.value
        if isinstance(render_mode, RenderMode):
            render_mode = render_mode.value
        if isinstance(post_process_type, PostProcessType):
            post_process_type = post_process_type.value

        payload = {
            "canvas": {
                "fill_type": fill_type,
                "render_mode": render_mode,
                "post_process": {"type": post_process_type},
            }
        }

        if post_process_type == "effect" and effect_params:
            payload["canvas"]["post_process"]["params"] = effect_params

        try:
            response = self.session.put(
                f"{self.stream_url}/start", headers=self.headers, json=payload
            )
            response.raise_for_status()

            response_data = response.json()
            session_id = (
                response_data.get("success", {}).get("data", {}).get("session_id")
            )

            if not session_id:
                raise APIError("No session ID received from device")

            self.is_streaming = True
            logger.info(f"Streaming session started with ID: {session_id}")
            return session_id

        except requests.RequestException as e:
            raise APIError(f"Failed to start streaming session: {str(e)}") from e

    def stop_streaming(self) -> None:
        """Stop the current streaming session.

        Raises:
            APIError: If the API request fails
        """
        if not self.is_streaming:
            logger.info("No active streaming session to stop")
            return

        try:
            response = self.session.put(f"{self.stream_url}/stop", headers=self.headers)
            response.raise_for_status()
            self.is_streaming = False
            logger.info("Streaming session stopped")

        except requests.RequestException as e:
            raise APIError(f"Failed to stop streaming session: {str(e)}") from e

    def send_frame(
        self,
        pixels: List[Tuple[int, int, int]],
        encoding: ContentEncoding = ContentEncoding.RAW,
        x: int = 0,
        y: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Send a single frame to the device.

        Args:
            pixels: List of RGB tuples (r,g,b) with values 0-255
            encoding: Content encoding type (default: RAW)
            x: X coordinate for the canvas area (default: 0)
            y: Y coordinate for the canvas area (default: 0)
            width: Width of the canvas area (default: device canvas width)
            height: Height of the canvas area (default: device canvas height)

        Raises:
            ValueError: If pixel data is invalid
            ProtocolError: If there's an issue with the protocol
        """
        if not self.is_streaming:
            raise ProtocolError("No active streaming session")

        if not self._socket:
            raise ProtocolError("Socket not initialized")

        if not self.session_id:
            raise ProtocolError("Missing session ID")

        width = width or self.canvas_size.get("width")
        height = height or self.canvas_size.get("height")

        if not width or not height:
            raise ValueError("Canvas dimensions not available")

        area = CanvasArea(x=x, y=y, width=width, height=height, pixels=pixels)
        area.validate()

        packet = self._create_lmsp_packet(area, encoding)
        self._send_udp_packet(packet)

    def send_frame_for_duration(
        self,
        pixels: List[Tuple[int, int, int]],
        duration: float,
        canvas_size: Optional[Tuple[int, int]] = None,
        auto_stop: bool = False,
    ) -> None:
        """Send a single frame and display it for a specific duration.

        Args:
            pixels: List of RGB tuples for the frame
            duration: Duration to display the frame in seconds
            canvas_size: Optional (width, height) tuple to override device size
            auto_stop: If True, stops the streaming session after the duration

        Raises:
            ValueError: If parameters are invalid
            ProtocolError: If there's an issue with the protocol
        """
        if not self.is_streaming:
            raise ProtocolError("No active streaming session")

        width, height = canvas_size or (
            self.canvas_size.get("width"),
            self.canvas_size.get("height"),
        )

        if not width or not height:
            raise ValueError("Canvas dimensions not available")

        self.send_frame(pixels=pixels, width=width, height=height)

        try:
            logger.info(f"Displaying frame for {duration} seconds")
            time.sleep(duration)
        except KeyboardInterrupt:
            logger.info("Frame display interrupted by user")

        if auto_stop and self.is_streaming:
            self.stop_streaming()

    def send_frames(
        self,
        frames: List[List[Tuple[int, int, int]]],
        fps: float = 30,
        frame_durations: Optional[List[float]] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
        loop: bool = False,
        loop_count: int = 0,  # 0 means infinite
    ) -> None:
        """Send multiple frames to the device at the specified timing.

        Args:
            frames: List of frames, where each frame is a list of RGB tuples
            fps: Frames per second (default: 30, ignored if frame_durations is provided)
            frame_durations: Optional list of durations for each frame in seconds
            canvas_size: Optional (width, height) tuple to override device size
            loop: Whether to loop the animation
            loop_count: Number of times to loop (0 = infinite)

        Raises:
            ValueError: If parameters are invalid
            ProtocolError: If there's an issue with the protocol
        """
        if not self.is_streaming:
            raise ProtocolError("No active streaming session")

        if not frames:
            raise ValueError("No frames provided")

        width, height = canvas_size or (
            self.canvas_size.get("width"),
            self.canvas_size.get("height"),
        )

        if not width or not height:
            raise ValueError("Canvas dimensions not available")

        if frame_durations:
            if len(frame_durations) != len(frames):
                raise ValueError(
                    f"Expected {len(frames)} frame durations, got {len(frame_durations)}"
                )

            timings = frame_durations
        else:
            frame_time = 1 / fps
            timings = [frame_time] * len(frames)

        running = True

        def stream_frames():
            nonlocal running

            iterations = 0
            while running and (loop_count == 0 or iterations < loop_count):
                for i, (frame_pixels, duration) in enumerate(zip(frames, timings)):
                    if not running:
                        break

                    try:
                        self.send_frame(pixels=frame_pixels, width=width, height=height)

                        time.sleep(duration)
                    except Exception as e:
                        logger.error(f"Error sending frame {i}: {str(e)}")
                        running = False
                        break

                if loop:
                    iterations += 1
                else:
                    break

        try:
            thread = threading.Thread(target=stream_frames)
            thread.daemon = True
            thread.start()

            while thread.is_alive():
                try:
                    thread.join(0.1)
                except KeyboardInterrupt:
                    logger.info("Streaming interrupted by user")
                    running = False
                    thread.join()
                    break

        except Exception as e:
            running = False
            raise ProtocolError(f"Error during frame streaming: {str(e)}") from e

    def _create_lmsp_packet(
        self, area: CanvasArea, encoding: ContentEncoding = ContentEncoding.RAW
    ) -> bytes:
        """Create an LMSP packet for the given canvas area.

        Args:
            area: CanvasArea object with pixel data
            encoding: Content encoding type

        Returns:
            Bytes containing the LMSP packet

        Raises:
            ProtocolError: If there's an issue creating the packet
        """
        try:
            if not self.session_id or len(self.session_id) != 32:
                raise ProtocolError(
                    "Session ID must be a 32-character hexadecimal string"
                )

            try:
                session_bytes = binascii.unhexlify(self.session_id)
            except binascii.Error:
                raise ProtocolError(f"Invalid session ID format: {self.session_id}")

            data_length = area.width * area.height * 3
            data_length_bytes = struct.pack("<H", data_length)

            header = (
                b"lmsp"  # Protocol name (4 bytes)
                + struct.pack("<H", 1)  # Version (2 bytes)
                + session_bytes  # Session ID (16 bytes)
                + struct.pack("B", encoding.value)  # Content encoding (1 byte)
                + b"\x00"  # Reserved (1 byte)
                + b"\x01"  # Canvas area count (1 byte)
                + b"\x00"  # Reserved (1 byte)
                + struct.pack("<H", area.x)  # Canvas area X (2 bytes)
                + struct.pack("<H", area.y)  # Canvas area Y (2 bytes)
                + struct.pack("<H", area.width)  # Canvas area width (2 bytes)
                + struct.pack("<H", area.height)  # Canvas area height (2 bytes)
                + data_length_bytes  # Data length (2 bytes)
            )

            pixel_data = bytearray()
            for pixel in area.pixels:
                pixel_data.extend([min(max(value, 0), 255) for value in pixel])

            return header + pixel_data

        except Exception as e:
            raise ProtocolError(f"Failed to create LMSP packet: {str(e)}") from e

    def _send_udp_packet(self, packet: bytes) -> None:
        """Send a UDP packet to the device.

        Args:
            packet: Bytes to send

        Raises:
            ProtocolError: If there's an issue sending the packet
        """
        if not self._socket:
            raise ProtocolError("Socket not initialized")

        try:
            self._socket.sendto(packet, (self.ip, self.port))
        except socket.error as e:
            raise ProtocolError(f"Failed to send UDP packet: {str(e)}") from e

    def send_solid_color(
        self, color: Tuple[int, int, int], duration: Optional[float] = None
    ) -> None:
        """Send a solid color frame to the device.

        Args:
            color: RGB tuple (r,g,b) with values 0-255
            duration: Optional duration in seconds to display the frame
        """
        width = self.canvas_size.get("width")
        height = self.canvas_size.get("height")

        if not width or not height:
            raise ValueError("Canvas dimensions not available")

        pixels = [color] * (width * height)

        if duration is not None:
            self.send_frame_for_duration(pixels=pixels, duration=duration)
        else:
            self.send_frame(pixels=pixels)

    def send_text(
        self,
        text: str,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        duration: Optional[float] = None,
        scroll: bool = True,
        scroll_speed: float = 0.05,  # Seconds per column shift
        center_vertical: bool = True,
        initial_pause: float = 1.0,
        end_pause: float = 1.0,
        clear_between_loops: bool = True,
        min_display_time: float = 2.0,
    ) -> None:
        """Send text to the device using a consistent bitmap font with support for descenders.

        Args:
            text: Text to display
            text_color: RGB color for text
            bg_color: RGB color for background
            duration: Optional duration in seconds to display the text
            scroll: Whether to scroll the text (if longer than display)
            scroll_speed: Speed of scrolling (seconds per column shift)
            center_vertical: Whether to center text vertically
            initial_pause: Time in seconds to pause on the first frame before scrolling
            end_pause: Time in seconds to pause on the empty screen at the end
            clear_between_loops: Whether to show a blank screen between loops
            min_display_time: Minimum time in seconds to display static text
        """
        FONT_WIDTH = 5
        FONT_HEIGHT = 7
        DESCENDER_HEIGHT = 8
        CHAR_SPACING = 1

        canvas_width = self.canvas_size.get("width")
        canvas_height = self.canvas_size.get("height")

        if not canvas_width or not canvas_height:
            raise ValueError("Canvas dimensions not available")

        text_width = len(text) * (FONT_WIDTH + CHAR_SPACING) - CHAR_SPACING

        needs_scrolling = text_width > canvas_width and scroll

        max_height = FONT_HEIGHT
        for char in text:
            if char in FONT_DESCENDERS:
                max_height = DESCENDER_HEIGHT
                break

        if needs_scrolling:
            text_bitmap = []
            for y in range(max_height):
                row = []
                for char in text:
                    has_descender = char in FONT_DESCENDERS
                    char_height = DESCENDER_HEIGHT if has_descender else FONT_HEIGHT

                    if has_descender:
                        char_bitmap = FONT_DESCENDERS.get(
                            char, FONT_5X7.get(" ", [0] * 7)
                        )
                    else:
                        char_bitmap = FONT_5X7.get(char, FONT_5X7.get(" ", [0] * 7))

                    for x in range(FONT_WIDTH):
                        if not has_descender and y >= FONT_HEIGHT:
                            bit = 0
                        elif y < char_height:
                            bit = (
                                1 if char_bitmap[y] & (1 << (FONT_WIDTH - 1 - x)) else 0
                            )
                        else:
                            bit = 0

                        row.append(bit)

                    if char != text[-1]:
                        for _ in range(CHAR_SPACING):
                            row.append(0)

                text_bitmap.append(row)

            frames = []
            frame_durations = []

            scroll_width = text_width + canvas_width

            if clear_between_loops:
                scroll_width += canvas_width

            for offset in range(scroll_width):
                frame = []

                y_offset = (canvas_height - max_height) // 2 if center_vertical else 0
                y_offset = max(0, y_offset)

                for _ in range(y_offset):
                    frame.extend([bg_color] * canvas_width)

                for y in range(min(max_height, canvas_height - y_offset)):
                    for x in range(canvas_width):
                        bitmap_x = x - (canvas_width - offset)

                        if 0 <= bitmap_x < len(text_bitmap[0]):
                            pixel = text_color if text_bitmap[y][bitmap_x] else bg_color
                        else:
                            pixel = bg_color

                        frame.append(pixel)

                remaining_rows = (
                    canvas_height - min(max_height, canvas_height - y_offset) - y_offset
                )
                for _ in range(remaining_rows):
                    frame.extend([bg_color] * canvas_width)

                frames.append(frame)

                if offset == 0:
                    frame_durations.append(initial_pause)
                elif offset == scroll_width - 1 and clear_between_loops:
                    frame_durations.append(end_pause)
                else:
                    frame_durations.append(scroll_speed)

            if duration is not None:
                total_animation_time = sum(frame_durations)
                loop_count = max(1, int(duration / total_animation_time))
                self.send_frames(
                    frames,
                    frame_durations=frame_durations,
                    loop=True,
                    loop_count=loop_count,
                )
            else:
                self.send_frames(frames, frame_durations=frame_durations, loop=True)

        else:
            frames = []
            frame_durations = []

            x_offset = (canvas_width - text_width) // 2
            y_offset = (canvas_height - max_height) // 2 if center_vertical else 0
            y_offset = max(0, y_offset)

            text_frame = []

            for _ in range(y_offset):
                text_frame.extend([bg_color] * canvas_width)

            for y in range(min(max_height, canvas_height - y_offset)):
                text_frame.extend([bg_color] * x_offset)

                x = x_offset
                for char in text:
                    has_descender = char in FONT_DESCENDERS
                    char_height = DESCENDER_HEIGHT if has_descender else FONT_HEIGHT

                    if has_descender:
                        char_bitmap = FONT_DESCENDERS.get(
                            char, FONT_5X7.get(" ", [0] * 7)
                        )
                    else:
                        char_bitmap = FONT_5X7.get(char, FONT_5X7.get(" ", [0] * 7))

                    for bit_pos in range(FONT_WIDTH):
                        if not has_descender and y >= FONT_HEIGHT:
                            bit = 0
                        elif y < char_height:
                            bit = (
                                1
                                if char_bitmap[y] & (1 << (FONT_WIDTH - 1 - bit_pos))
                                else 0
                            )
                        else:
                            bit = 0

                        pixel = text_color if bit else bg_color
                        text_frame.append(pixel)
                        x += 1

                    if char != text[-1]:
                        for _ in range(CHAR_SPACING):
                            text_frame.append(bg_color)
                            x += 1

                remaining = canvas_width - x
                text_frame.extend([bg_color] * remaining)

            remaining_rows = (
                canvas_height - min(max_height, canvas_height - y_offset) - y_offset
            )
            for _ in range(remaining_rows):
                text_frame.extend([bg_color] * canvas_width)

            blank_frame = [bg_color] * (canvas_width * canvas_height)

            frames.append(text_frame)
            frame_durations.append(initial_pause)

            if duration is not None:
                display_time = duration - initial_pause
                if clear_between_loops:
                    display_time -= end_pause

                display_time = max(min_display_time, display_time)

                frames.append(text_frame)
                frame_durations.append(display_time)

                if clear_between_loops:
                    frames.append(blank_frame)
                    frame_durations.append(end_pause)

                self.send_frames(frames, frame_durations=frame_durations, loop=False)
            else:
                frames.append(text_frame)
                frame_durations.append(min_display_time)

                if clear_between_loops:
                    frames.append(blank_frame)
                    frame_durations.append(end_pause)

                self.send_frames(frames, frame_durations=frame_durations, loop=True)

    # Region management methods

    def create_region(
        self, x: int, y: int, width: int, height: int, region_id: Optional[str] = None
    ) -> str:
        """Create a new screen region.

        Args:
            x: X coordinate of the region
            y: Y coordinate of the region
            width: Width of the region
            height: Height of the region
            region_id: Optional ID for the region

        Returns:
            ID of the created region

        Raises:
            ValueError: If region parameters are invalid
        """
        if not self.region_manager:
            raise LMStreamError("Region manager not initialized")

        return self.region_manager.create_region(x, y, width, height, region_id)

    def create_grid(self, rows: int, cols: int) -> List[str]:
        """Create a grid of equal-sized regions.

        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid

        Returns:
            List of region IDs, ordered row by row
        """
        if not self.region_manager:
            raise LMStreamError("Region manager not initialized")

        return self.region_manager.create_grid(rows, cols)

    def delete_region(self, region_id: str) -> bool:
        """Delete a region.

        Args:
            region_id: ID of the region to delete

        Returns:
            True if the region was deleted, False if not found
        """
        if not self.region_manager:
            raise LMStreamError("Region manager not initialized")

        return self.region_manager.delete_region(region_id)

    def clear_regions(self) -> None:
        """Delete all regions."""
        if not self.region_manager:
            raise LMStreamError("Region manager not initialized")

        self.region_manager.clear_regions()

    def get_region(self, region_id: str) -> Optional[ScreenRegion]:
        """Get a region by ID.

        Args:
            region_id: ID of the region to get

        Returns:
            ScreenRegion object or None if not found
        """
        if not self.region_manager:
            raise LMStreamError("Region manager not initialized")

        return self.region_manager.get_region(region_id)

    def set_region_text(
        self,
        region_id: str,
        text: str,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        scroll: bool = True,
        scroll_speed: float = 0.05,
        center_vertical: bool = True,
        initial_pause: float = 1.0,
        end_pause: float = 1.0,
        clear_between_loops: bool = True,
    ) -> None:
        """Set text content for a region.

        Args:
            region_id: ID of the region to update
            text: Text to display
            text_color: RGB color for text
            bg_color: RGB color for background
            scroll: Whether to scroll the text (if longer than display)
            scroll_speed: Speed of scrolling (seconds per column shift)
            center_vertical: Whether to center text vertically
            initial_pause: Time to pause before scrolling starts
            end_pause: Time to pause at the end of scrolling
            clear_between_loops: Whether to show blank screen between loops
        """
        region = self.get_region(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")

        region.set_text(
            text=text,
            text_color=text_color,
            bg_color=bg_color,
            scroll=scroll,
            scroll_speed=scroll_speed,
            center_vertical=center_vertical,
            initial_pause=initial_pause,
            end_pause=end_pause,
            clear_between_loops=clear_between_loops,
        )

    def set_region_color(
        self,
        region_id: str,
        color: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        gradient_horizontal: bool = True,
    ) -> None:
        """Set color content for a region.

        Args:
            region_id: ID of the region to update
            color: Single RGB tuple for solid color or list of RGB tuples for gradient
            gradient_horizontal: If True, gradient runs horizontally, else vertically
        """
        region = self.get_region(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")

        region.set_color(color=color, gradient_horizontal=gradient_horizontal)

    def set_region_animation(
        self,
        region_id: str,
        frames: List[List[Tuple[int, int, int]]],
        frame_durations: Union[List[float], float] = 0.1,
        loop: bool = True,
    ) -> None:
        """Set animation content for a region.

        Args:
            region_id: ID of the region to update
            frames: List of frames, each frame is a list of RGB tuples
            frame_durations: Duration for each frame or list of durations
            loop: Whether to loop the animation
        """
        region = self.get_region(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")

        region.set_animation(frames=frames, frame_durations=frame_durations, loop=loop)

    def clear_region_content(self, region_id: str) -> None:
        """Clear content from a region.

        Args:
            region_id: ID of the region to clear
        """
        region = self.get_region(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")

        region.clear_content()

    def show_region(self, region_id: str) -> None:
        """Show a region.

        Args:
            region_id: ID of the region to show
        """
        region = self.get_region(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")

        region.active = True

    def hide_region(self, region_id: str) -> None:
        """Hide a region.

        Args:
            region_id: ID of the region to hide
        """
        region = self.get_region(region_id)
        if not region:
            raise ValueError(f"Region {region_id} not found")

        region.active = False

    def _on_regions_updated(self, updated_region_ids: List[str]) -> None:
        """Handle region updates.

        Args:
            updated_region_ids: List of IDs of regions that were updated
        """
        if not self.is_streaming or not self.region_manager:
            return

        canvas_frame = self.region_manager.get_canvas_frame()

        try:
            self.send_frame(
                pixels=canvas_frame,
                width=self.canvas_size.get("width"),
                height=self.canvas_size.get("height"),
            )
        except Exception as e:
            logger.error(f"Error sending updated frame: {str(e)}")

    def _start_update_thread(self) -> None:
        """Start the thread that updates regions."""
        if self._update_thread and self._update_thread.is_alive():
            return

        self._update_running = True

        def update_loop():
            last_update_time = time.time()

            while self._update_running and self.is_streaming:
                current_time = time.time()
                delta_time = current_time - last_update_time
                last_update_time = current_time

                if self.region_manager:
                    try:
                        self.region_manager.update_regions(delta_time)
                    except Exception as e:
                        logger.error(f"Error updating regions: {str(e)}")

                # Sleep to maintain update interval
                sleep_time = max(
                    0.001, self.update_interval - (time.time() - current_time)
                )
                time.sleep(sleep_time)

        self._update_thread = threading.Thread(target=update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()

    def _stop_update_thread(self) -> None:
        """Stop the update thread."""
        self._update_running = False

        if self._update_thread and self._update_thread.is_alive():
            try:
                self._update_thread.join(timeout=1.0)
            except Exception:
                pass

            self._update_thread = None
