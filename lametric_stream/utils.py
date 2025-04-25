"""
Utility functions for the LaMetric Stream library.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Union

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


class RegionContent:
    """Base class for content that can be displayed in a screen region."""

    def get_current_frame(self) -> List[Tuple[int, int, int]]:
        """Get the current frame of content.

        Returns:
            List of RGB tuples representing the current frame
        """
        raise NotImplementedError("Subclasses must implement get_current_frame")

    def update(self, delta_time: float) -> bool:
        """Update the content state.

        Args:
            delta_time: Time in seconds since last update

        Returns:
            True if the content has changed and needs redrawing
        """
        return False

    def reset(self) -> None:
        """Reset the content to its initial state."""
        pass


class TextContent(RegionContent):
    """Scrollable text content for a screen region."""

    def __init__(
        self,
        text: str,
        region_width: int,
        region_height: int,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        scroll: bool = True,
        scroll_speed: float = 0.05,
        center_vertical: bool = True,
        initial_pause: float = 1.0,
        end_pause: float = 1.0,
        clear_between_loops: bool = True,
    ):
        """Initialize text content.

        Args:
            text: Text to display
            region_width: Width of the region in pixels
            region_height: Height of the region in pixels
            text_color: RGB color for text
            bg_color: RGB color for background
            scroll: Whether to scroll the text (if longer than display)
            scroll_speed: Speed of scrolling (seconds per column shift)
            center_vertical: Whether to center text vertically
            initial_pause: Time to pause before scrolling starts
            end_pause: Time to pause at the end of scrolling
            clear_between_loops: Whether to show blank screen between loops
        """
        self.text = text
        self.region_width = region_width
        self.region_height = region_height
        self.text_color = text_color
        self.bg_color = bg_color
        self.scroll = scroll
        self.scroll_speed = scroll_speed
        self.center_vertical = center_vertical
        self.initial_pause = initial_pause
        self.end_pause = end_pause
        self.clear_between_loops = clear_between_loops

        # Font sizes and properties
        self.FONT_WIDTH = 5
        self.FONT_HEIGHT = 7
        self.DESCENDER_HEIGHT = 8
        self.CHAR_SPACING = 1

        # Calculate text properties
        self.text_width = (
            len(self.text) * (self.FONT_WIDTH + self.CHAR_SPACING) - self.CHAR_SPACING
        )
        self.needs_scrolling = self.text_width > self.region_width and self.scroll

        # Determine if any characters need descenders
        self.max_height = self.FONT_HEIGHT
        self.has_descenders = False

        # Reset to initial state
        self.reset()

        # Pre-render the bitmap representation of the text
        self._render_text_bitmap()

    def _render_text_bitmap(self):
        """Render the text as a bitmap representation."""
        from lametric_stream.fonts import FONT_5X7, FONT_DESCENDERS

        self.text_bitmap = []

        # Check for descenders
        for char in self.text:
            if char in FONT_DESCENDERS:
                self.max_height = self.DESCENDER_HEIGHT
                self.has_descenders = True
                break

        # Create bitmap representation
        for y in range(self.max_height):
            row = []
            for char in self.text:
                has_descender = char in FONT_DESCENDERS
                char_height = (
                    self.DESCENDER_HEIGHT if has_descender else self.FONT_HEIGHT
                )

                if has_descender:
                    char_bitmap = FONT_DESCENDERS.get(char, FONT_5X7.get(" ", [0] * 7))
                else:
                    char_bitmap = FONT_5X7.get(char, FONT_5X7.get(" ", [0] * 7))

                for x in range(self.FONT_WIDTH):
                    if not has_descender and y >= self.FONT_HEIGHT:
                        bit = 0
                    elif y < char_height:
                        bit = (
                            1
                            if char_bitmap[y] & (1 << (self.FONT_WIDTH - 1 - x))
                            else 0
                        )
                    else:
                        bit = 0

                    row.append(bit)

                if char != self.text[-1]:
                    for _ in range(self.CHAR_SPACING):
                        row.append(0)

            self.text_bitmap.append(row)

        # Calculate total scroll width
        self.scroll_width = self.text_width + self.region_width
        if self.clear_between_loops:
            self.scroll_width += self.region_width

    def reset(self) -> None:
        """Reset text animation to initial state."""
        self.offset = 0
        self.current_time = 0
        self.pause_timer = self.initial_pause if self.needs_scrolling else 0
        self.state = "initial_pause" if self.pause_timer > 0 else "scrolling"

    def update(self, delta_time: float) -> bool:
        """Update text animation.

        Args:
            delta_time: Time in seconds since last update

        Returns:
            True if the frame needs to be redrawn
        """
        if not self.needs_scrolling:
            return False

        self.current_time += delta_time
        changed = False

        if self.state == "initial_pause":
            self.pause_timer -= delta_time
            if self.pause_timer <= 0:
                self.state = "scrolling"
                self.pause_timer = 0

        elif self.state == "scrolling":
            time_per_column = self.scroll_speed
            columns_to_move = int(
                (self.current_time - self.pause_timer) / time_per_column
            )

            if columns_to_move > 0:
                self.current_time -= columns_to_move * time_per_column
                old_offset = self.offset
                self.offset = (self.offset + columns_to_move) % self.scroll_width

                if (
                    old_offset < self.scroll_width - 1
                    and self.offset >= self.scroll_width - 1
                ):
                    self.state = "end_pause"
                    self.pause_timer = self.end_pause

                changed = True

        elif self.state == "end_pause":
            self.pause_timer -= delta_time
            if self.pause_timer <= 0:
                self.offset = 0
                self.state = "initial_pause" if self.initial_pause > 0 else "scrolling"
                self.pause_timer = self.initial_pause
                changed = True

        return changed

    def get_current_frame(self) -> List[Tuple[int, int, int]]:
        """Get the current frame of the text content.

        Returns:
            List of RGB tuples representing pixels
        """
        frame = []

        if self.needs_scrolling:
            y_offset = (
                (self.region_height - self.max_height) // 2
                if self.center_vertical
                else 0
            )
            y_offset = max(0, y_offset)

            # Add top padding rows
            for _ in range(y_offset):
                frame.extend([self.bg_color] * self.region_width)

            # Add text rows
            for y in range(min(self.max_height, self.region_height - y_offset)):
                for x in range(self.region_width):
                    bitmap_x = x - (self.region_width - self.offset)

                    if 0 <= bitmap_x < len(self.text_bitmap[0]):
                        pixel = (
                            self.text_color
                            if self.text_bitmap[y][bitmap_x]
                            else self.bg_color
                        )
                    else:
                        pixel = self.bg_color

                    frame.append(pixel)

            # Add bottom padding rows
            remaining_rows = (
                self.region_height
                - min(self.max_height, self.region_height - y_offset)
                - y_offset
            )
            for _ in range(remaining_rows):
                frame.extend([self.bg_color] * self.region_width)

        else:
            # Static text that fits without scrolling
            x_offset = (self.region_width - self.text_width) // 2
            y_offset = (
                (self.region_height - self.max_height) // 2
                if self.center_vertical
                else 0
            )
            y_offset = max(0, y_offset)

            # Add top padding rows
            for _ in range(y_offset):
                frame.extend([self.bg_color] * self.region_width)

            # Add text rows
            for y in range(min(self.max_height, self.region_height - y_offset)):
                row_pixels = []

                # Left padding
                row_pixels.extend([self.bg_color] * x_offset)

                # Text pixels
                x = x_offset
                for char_idx, char in enumerate(self.text):
                    from lametric_stream.fonts import FONT_5X7, FONT_DESCENDERS

                    has_descender = char in FONT_DESCENDERS
                    char_height = (
                        self.DESCENDER_HEIGHT if has_descender else self.FONT_HEIGHT
                    )

                    if has_descender:
                        char_bitmap = FONT_DESCENDERS.get(
                            char, FONT_5X7.get(" ", [0] * 7)
                        )
                    else:
                        char_bitmap = FONT_5X7.get(char, FONT_5X7.get(" ", [0] * 7))

                    for bit_pos in range(self.FONT_WIDTH):
                        if not has_descender and y >= self.FONT_HEIGHT:
                            bit = 0
                        elif y < char_height:
                            bit = (
                                1
                                if char_bitmap[y]
                                & (1 << (self.FONT_WIDTH - 1 - bit_pos))
                                else 0
                            )
                        else:
                            bit = 0

                        pixel = self.text_color if bit else self.bg_color
                        row_pixels.append(pixel)
                        x += 1

                    # Add spacing between characters
                    if char_idx < len(self.text) - 1:
                        for _ in range(self.CHAR_SPACING):
                            row_pixels.append(self.bg_color)
                            x += 1

                # Right padding
                remaining = self.region_width - x
                row_pixels.extend([self.bg_color] * remaining)

                frame.extend(row_pixels)

            # Add bottom padding rows
            remaining_rows = (
                self.region_height
                - min(self.max_height, self.region_height - y_offset)
                - y_offset
            )
            for _ in range(remaining_rows):
                frame.extend([self.bg_color] * self.region_width)

        return frame


class ColorContent(RegionContent):
    """Solid color or gradient content for a screen region."""

    def __init__(
        self,
        region_width: int,
        region_height: int,
        color: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        gradient_horizontal: bool = True,
    ):
        """Initialize color content.

        Args:
            region_width: Width of the region in pixels
            region_height: Height of the region in pixels
            color: Single RGB tuple for solid color or list of RGB tuples for gradient
            gradient_horizontal: If True, gradient runs horizontally, else vertically
        """
        self.region_width = region_width
        self.region_height = region_height

        if isinstance(color, tuple) and len(color) == 3:
            self.colors = [color]
            self.is_gradient = False
        elif isinstance(color, list) and len(color) > 1:
            self.colors = color
            self.is_gradient = True
        else:
            raise ValueError("Color must be a single RGB tuple or a list of RGB tuples")

        self.gradient_horizontal = gradient_horizontal
        self._render_frame()

    def _render_frame(self):
        """Render the frame with solid color or gradient."""
        self.frame = []

        if not self.is_gradient:
            # Solid color
            self.frame = [self.colors[0]] * (self.region_width * self.region_height)
        else:
            # Gradient
            if self.gradient_horizontal:
                for y in range(self.region_height):
                    for x in range(self.region_width):
                        t = x / (self.region_width - 1) if self.region_width > 1 else 0
                        color = self._interpolate_color(t)
                        self.frame.append(color)
            else:
                for y in range(self.region_height):
                    t = y / (self.region_height - 1) if self.region_height > 1 else 0
                    color = self._interpolate_color(t)
                    self.frame.extend([color] * self.region_width)

    def _interpolate_color(self, t: float) -> Tuple[int, int, int]:
        """Interpolate between colors in a gradient.

        Args:
            t: Value between 0 and 1 representing position in gradient

        Returns:
            Interpolated RGB color
        """
        if t <= 0:
            return self.colors[0]
        if t >= 1:
            return self.colors[-1]

        segment_count = len(self.colors) - 1
        segment = int(t * segment_count)
        segment_t = (t * segment_count) - segment

        start_color = self.colors[segment]
        end_color = self.colors[segment + 1]

        r = int(start_color[0] + segment_t * (end_color[0] - start_color[0]))
        g = int(start_color[1] + segment_t * (end_color[1] - start_color[1]))
        b = int(start_color[2] + segment_t * (end_color[2] - start_color[2]))

        return (r, g, b)

    def get_current_frame(self) -> List[Tuple[int, int, int]]:
        """Get the current frame.

        Returns:
            List of RGB tuples
        """
        return self.frame

    def update(self, delta_time: float) -> bool:
        """Update the content (no animation by default).

        Args:
            delta_time: Time in seconds since last update

        Returns:
            True if the frame needs to be redrawn
        """
        return False


class AnimationContent(RegionContent):
    """Animated content for a screen region."""

    def __init__(
        self,
        region_width: int,
        region_height: int,
        frames: List[List[Tuple[int, int, int]]],
        frame_durations: Union[List[float], float] = 0.1,
        loop: bool = True,
    ):
        """Initialize animation content.

        Args:
            region_width: Width of the region in pixels
            region_height: Height of the region in pixels
            frames: List of frames, each frame is a list of RGB tuples
            frame_durations: Duration for each frame or list of durations
            loop: Whether to loop the animation
        """
        self.region_width = region_width
        self.region_height = region_height
        self.frames = frames
        self.loop = loop

        if isinstance(frame_durations, (int, float)):
            self.frame_durations = [float(frame_durations)] * len(frames)
        else:
            if len(frame_durations) != len(frames):
                raise ValueError(
                    f"Expected {len(frames)} frame durations, got {len(frame_durations)}"
                )
            self.frame_durations = list(map(float, frame_durations))

        self.reset()

    def reset(self) -> None:
        """Reset animation to initial state."""
        self.current_frame_index = 0
        self.elapsed_time = 0

    def update(self, delta_time: float) -> bool:
        """Update animation state.

        Args:
            delta_time: Time in seconds since last update

        Returns:
            True if the frame changed
        """
        if len(self.frames) <= 1:
            return False

        old_frame_index = self.current_frame_index

        self.elapsed_time += delta_time

        # Check if we need to advance to the next frame
        if self.elapsed_time >= self.frame_durations[self.current_frame_index]:
            self.elapsed_time -= self.frame_durations[self.current_frame_index]
            self.current_frame_index += 1

            # Handle looping or end of animation
            if self.current_frame_index >= len(self.frames):
                if self.loop:
                    self.current_frame_index = 0
                else:
                    self.current_frame_index = len(self.frames) - 1

        return old_frame_index != self.current_frame_index

    def get_current_frame(self) -> List[Tuple[int, int, int]]:
        """Get the current frame of the animation.

        Returns:
            List of RGB tuples
        """
        return self.frames[self.current_frame_index]


@dataclass
class ScreenRegion:
    """Represents a region of the screen that can be updated independently."""

    # Position and dimensions
    x: int
    y: int
    width: int
    height: int

    # Content management
    content: Optional[RegionContent] = None
    active: bool = True
    id: str = field(default_factory=lambda: f"region_{id(object())}")

    def __post_init__(self):
        """Validate region after initialization."""
        if not (0 <= self.x and 0 <= self.y):
            raise ValueError(f"Invalid coordinates: ({self.x}, {self.y})")

        if not (0 < self.width and 0 < self.height):
            raise ValueError(f"Invalid dimensions: {self.width}x{self.height}")

    def set_content(self, content: RegionContent) -> None:
        """Set the content for this region.

        Args:
            content: RegionContent object to display in this region
        """
        self.content = content

    def clear_content(self) -> None:
        """Clear the content from this region."""
        self.content = None

    def set_text(
        self,
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
        """Set text content for this region.

        Args:
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
        self.content = TextContent(
            text=text,
            region_width=self.width,
            region_height=self.height,
            text_color=text_color,
            bg_color=bg_color,
            scroll=scroll,
            scroll_speed=scroll_speed,
            center_vertical=center_vertical,
            initial_pause=initial_pause,
            end_pause=end_pause,
            clear_between_loops=clear_between_loops,
        )

    def set_color(
        self,
        color: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        gradient_horizontal: bool = True,
    ) -> None:
        """Set color content for this region.

        Args:
            color: Single RGB tuple for solid color or list of RGB tuples for gradient
            gradient_horizontal: If True, gradient runs horizontally, else vertically
        """
        self.content = ColorContent(
            region_width=self.width,
            region_height=self.height,
            color=color,
            gradient_horizontal=gradient_horizontal,
        )

    def set_animation(
        self,
        frames: List[List[Tuple[int, int, int]]],
        frame_durations: Union[List[float], float] = 0.1,
        loop: bool = True,
    ) -> None:
        """Set animation content for this region.

        Args:
            frames: List of frames, each frame is a list of RGB tuples
            frame_durations: Duration for each frame or list of durations
            loop: Whether to loop the animation
        """
        self.content = AnimationContent(
            region_width=self.width,
            region_height=self.height,
            frames=frames,
            frame_durations=frame_durations,
            loop=loop,
        )

    def update(self, delta_time: float) -> bool:
        """Update the region's content.

        Args:
            delta_time: Time in seconds since last update

        Returns:
            True if the region's content has changed
        """
        if not self.active or not self.content:
            return False

        return self.content.update(delta_time)

    def get_frame(self) -> List[Tuple[int, int, int]]:
        """Get the current frame for this region.

        Returns:
            List of RGB tuples representing current frame, or empty list if no content
        """
        if not self.active or not self.content:
            return [(0, 0, 0)] * (self.width * self.height)

        return self.content.get_current_frame()


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


class RegionManager:
    """Manages multiple screen regions."""

    def __init__(self, canvas_width: int, canvas_height: int):
        """Initialize the region manager.

        Args:
            canvas_width: Width of the entire canvas
            canvas_height: Height of the entire canvas
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.regions: Dict[str, ScreenRegion] = {}
        self.update_callback: Optional[Callable[[List[str]], None]] = None

    def create_region(
        self, x: int, y: int, width: int, height: int, region_id: Optional[str] = None
    ) -> str:
        """Create a new screen region.

        Args:
            x: X coordinate of the region
            y: Y coordinate of the region
            width: Width of the region
            height: Height of the region
            region_id: Optional ID for the region (generated if not provided)

        Returns:
            ID of the created region

        Raises:
            ValueError: If region parameters are invalid or if region overlaps
        """
        # Validate coordinates
        if (
            x < 0
            or y < 0
            or x + width > self.canvas_width
            or y + height > self.canvas_height
        ):
            raise ValueError(
                f"Region ({x},{y},{width},{height}) exceeds canvas dimensions"
            )

        # Check for overlaps
        for region in self.regions.values():
            if (
                x < region.x + region.width
                and x + width > region.x
                and y < region.y + region.height
                and y + height > region.y
            ):
                raise ValueError(
                    f"Region ({x},{y},{width},{height}) overlaps with existing region"
                )

        # Create region
        region = ScreenRegion(x=x, y=y, width=width, height=height)

        # Use provided ID or keep generated one
        if region_id:
            region.id = region_id

        self.regions[region.id] = region
        return region.id

    def create_grid(self, rows: int, cols: int) -> List[str]:
        """Create a grid of equal-sized regions.

        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid

        Returns:
            List of region IDs, ordered row by row

        Raises:
            ValueError: If rows or columns are invalid
        """
        if rows <= 0 or cols <= 0:
            raise ValueError("Rows and columns must be positive")

        # Clear existing regions
        self.regions.clear()

        # Calculate region dimensions
        region_width = self.canvas_width // cols
        region_height = self.canvas_height // rows

        # Create regions
        region_ids = []
        for r in range(rows):
            for c in range(cols):
                x = c * region_width
                y = r * region_height

                # Adjust last row/column to fill canvas
                w = region_width if c < cols - 1 else self.canvas_width - x
                h = region_height if r < rows - 1 else self.canvas_height - y

                region_id = f"r{r}c{c}"
                self.create_region(x, y, w, h, region_id)
                region_ids.append(region_id)

        return region_ids

    def delete_region(self, region_id: str) -> bool:
        """Delete a region.

        Args:
            region_id: ID of the region to delete

        Returns:
            True if the region was deleted, False if not found
        """
        if region_id in self.regions:
            del self.regions[region_id]
            return True
        return False

    def clear_regions(self) -> None:
        """Delete all regions."""
        self.regions.clear()

    def get_region(self, region_id: str) -> Optional[ScreenRegion]:
        """Get a region by ID.

        Args:
            region_id: ID of the region to get

        Returns:
            ScreenRegion object or None if not found
        """
        return self.regions.get(region_id)

    def update_regions(self, delta_time: float) -> List[str]:
        """Update all regions.

        Args:
            delta_time: Time in seconds since last update

        Returns:
            List of IDs of regions that changed
        """
        changed_regions = []

        for region_id, region in list(self.regions.items()):
            if region.update(delta_time):
                changed_regions.append(region_id)

        if changed_regions and self.update_callback:
            self.update_callback(changed_regions)

        return changed_regions

    def get_canvas_frame(self) -> List[Tuple[int, int, int]]:
        """Generate a complete frame for the entire canvas.

        Returns:
            List of RGB tuples for the canvas pixels
        """
        # Create an empty canvas filled with black
        canvas = [(0, 0, 0)] * (self.canvas_width * self.canvas_height)

        # Fill in each region
        for region in self.regions.values():
            if not region.active:
                continue

            region_frame = region.get_frame()

            # Map region pixels to canvas
            for y in range(region.height):
                for x in range(region.width):
                    # Calculate indices
                    region_idx = y * region.width + x
                    canvas_idx = (region.y + y) * self.canvas_width + (region.x + x)

                    # Check bounds
                    if 0 <= region_idx < len(region_frame) and 0 <= canvas_idx < len(
                        canvas
                    ):
                        canvas[canvas_idx] = region_frame[region_idx]

        return canvas

    def set_update_callback(self, callback: Callable[[List[str]], None]) -> None:
        """Set a callback to be called when regions are updated.

        Args:
            callback: Function to call with list of updated region IDs
        """
        self.update_callback = callback


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
