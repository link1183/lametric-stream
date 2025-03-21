"""
Animation-related code for the LaMetric Stream library.
"""

from enum import Enum
from typing import List, Tuple, Optional


class AnimationType(Enum):
    """Animation types supported by the library."""

    BLINK = "blink"
    WAVE = "wave"
    PULSE = "pulse"


def create_blink_animation(
    colors: List[Tuple[int, int, int]], width: int, height: int, speed: float = 1.0
) -> Tuple[List[List[Tuple[int, int, int]]], List[float]]:
    """Create frames for a blink animation.

    Args:
        colors: List of RGB colors to blink between
        width: Width of the frame
        height: Height of the frame
        speed: Speed multiplier for the animation

    Returns:
        Tuple containing list of frames and list of frame durations
    """
    # Simple blink animation alternating between colors
    frames = []
    frame_durations = []

    for color in colors:
        # Full frame of this color
        frame = [color] * (width * height)
        frames.append(frame)
        frame_durations.append(speed)

    return frames, frame_durations


def create_pulse_animation(
    colors: List[Tuple[int, int, int]],
    width: int,
    height: int,
    speed: float = 1.0,
    steps: int = 10,
) -> Tuple[List[List[Tuple[int, int, int]]], List[float]]:
    """Create frames for a pulse animation.

    Args:
        colors: List of RGB colors to pulse between
        width: Width of the frame
        height: Height of the frame
        speed: Speed multiplier for the animation
        steps: Number of transition steps between colors

    Returns:
        Tuple containing list of frames and list of frame durations
    """
    frames = []
    frame_durations = []

    # Pulse animation that fades between colors
    for i in range(len(colors)):
        color1 = colors[i]
        color2 = colors[(i + 1) % len(colors)]

        # Create transition frames between colors
        for step in range(steps):
            t = step / (steps - 1)
            r = int(color1[0] * (1 - t) + color2[0] * t)
            g = int(color1[1] * (1 - t) + color2[1] * t)
            b = int(color1[2] * (1 - t) + color2[2] * t)

            frame = [(r, g, b)] * (width * height)
            frames.append(frame)
            frame_durations.append(0.1 / speed)  # 0.1 seconds per transition frame

    return frames, frame_durations


def create_wave_animation(
    colors: List[Tuple[int, int, int]], width: int, height: int, speed: float = 1.0
) -> Tuple[List[List[Tuple[int, int, int]]], List[float]]:
    """Create frames for a wave animation.

    Args:
        colors: List of RGB colors to use in the wave
        width: Width of the frame
        height: Height of the frame
        speed: Speed multiplier for the animation

    Returns:
        Tuple containing list of frames and list of frame durations
    """
    frames = []
    frame_durations = []

    # Wave animation moving across the display
    for offset in range(width * 2):
        frame = []
        for _ in range(height):
            for x in range(width):
                # Create a wave pattern based on position
                color_idx = (x + offset) % len(colors)
                frame.append(colors[color_idx])

        frames.append(frame)
        frame_durations.append(speed)

    return frames, frame_durations


def create_animation(
    animation_type: AnimationType,
    colors: List[Tuple[int, int, int]],
    width: int,
    height: int,
    speed: float = 1.0,
    duration: Optional[float] = None,
) -> Tuple[List[List[Tuple[int, int, int]]], List[float]]:
    """Create frames and durations for the specified animation type.

    Args:
        animation_type: Type of animation
        colors: List of RGB colors to use in the animation
        width: Width of the frame
        height: Height of the frame
        speed: Speed multiplier for the animation
        duration: Optional total duration in seconds

    Returns:
        Tuple containing list of frames and list of frame durations
    """
    if not colors:
        raise ValueError("Colors list cannot be empty")

    if animation_type == AnimationType.BLINK:
        frames, frame_durations = create_blink_animation(colors, width, height, speed)
    elif animation_type == AnimationType.PULSE:
        frames, frame_durations = create_pulse_animation(colors, width, height, speed)
    elif animation_type == AnimationType.WAVE:
        frames, frame_durations = create_wave_animation(colors, width, height, speed)

    # Adjust frame durations to match total duration if specified
    if duration is not None:
        total_frames = len(frames)
        frame_duration = duration / total_frames
        frame_durations = [frame_duration] * total_frames

    return frames, frame_durations
