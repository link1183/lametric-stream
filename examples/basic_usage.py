#!/usr/bin/env python3
"""
Basic usage examples for the LaMetric Stream library.
"""

import os
from dotenv import load_dotenv
from lametric_stream import LMStream, AnimationType


# Load API key and IP address from environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
IP_ADDRESS = os.getenv("IP_ADDRESS")

if not API_KEY or not IP_ADDRESS:
    print("Error: API_KEY and IP_ADDRESS must be set in the .env file")
    print("Create a .env file with the following content:")
    print("API_KEY=your_api_key")
    print("IP_ADDRESS=your_device_ip")
    exit(1)


def example_text():
    """Display different text examples."""
    with LMStream(api_key=API_KEY, ip=IP_ADDRESS) as stream:
        # Simple text display
        stream.send_text(
            "Hello LaMetric!", text_color=(255, 255, 0), duration=3  # Yellow text
        )

        # Text with custom settings
        stream.send_text(
            "Scrolling text with custom settings",
            text_color=(0, 255, 255),  # Cyan text
            bg_color=(0, 0, 128),  # Dark blue background
            scroll_speed=0.03,  # Faster scrolling
            initial_pause=2.0,  # Longer initial pause
            end_pause=1.5,  # Custom end pause
            duration=10,
        )

        # Static text (won't scroll)
        stream.send_text(
            "OK",
            text_color=(0, 255, 0),  # Green text
            bg_color=(0, 0, 0),  # Black background
            scroll=False,  # Disable scrolling
            min_display_time=3.0,  # Display for at least 3 seconds
            duration=5,
        )


def example_colors():
    """Display color examples."""
    with LMStream(api_key=API_KEY, ip=IP_ADDRESS) as stream:
        # Solid color
        stream.send_solid_color((255, 0, 0), duration=2)  # Red

        # Gradients
        stream.send_gradient(
            start_color=(0, 0, 255),  # Blue
            end_color=(0, 255, 0),  # Green
            horizontal=True,  # Left to right gradient
            duration=3,
        )

        stream.send_gradient(
            start_color=(255, 0, 0),  # Red
            end_color=(0, 0, 255),  # Blue
            horizontal=False,  # Top to bottom gradient
            duration=3,
        )


def example_animations():
    """Display animation examples."""
    with LMStream(api_key=API_KEY, ip=IP_ADDRESS) as stream:
        # Blink animation
        stream.send_animation(
            animation_type=AnimationType.BLINK,
            colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],  # Red, Green, Blue
            speed=0.5,  # Slower speed (higher value = slower)
            duration=5,
        )

        # Pulse animation
        stream.send_animation(
            animation_type=AnimationType.PULSE,
            colors=[(255, 0, 0), (0, 0, 255)],  # Pulse between red and blue
            speed=1.0,
            duration=5,
        )

        # Wave animation
        stream.send_animation(
            animation_type=AnimationType.WAVE,
            colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            speed=0.3,  # Faster
            duration=5,
        )


def example_custom_frames():
    """Example of creating and sending custom frames."""
    with LMStream(api_key=API_KEY, ip=IP_ADDRESS) as stream:
        # Get canvas dimensions
        width = stream.canvas_size.get("width")
        height = stream.canvas_size.get("height")

        # Create a checkerboard pattern
        checkerboard = []
        for y in range(height):
            for x in range(width):
                if (x + y) % 2 == 0:
                    checkerboard.append((255, 255, 255))  # White
                else:
                    checkerboard.append((0, 0, 0))  # Black

        # Send the custom frame
        stream.send_frame_for_duration(checkerboard, duration=3)

        # Create a simple animation (moving line)
        frames = []
        for x in range(width):
            frame = []
            for y in range(height):
                for fx in range(width):
                    if fx == x:
                        frame.append((255, 255, 255))  # White line
                    else:
                        frame.append((0, 0, 0))  # Black background
            frames.append(frame)

        # Send the animation frames
        stream.send_frames(
            frames=frames,
            fps=10,  # 10 frames per second
            loop=True,  # Loop the animation
            loop_count=3,  # Loop 3 times
        )


if __name__ == "__main__":
    print("Running LaMetric Stream examples...")

    try:
        print("\nExample 1: Text Display")
        example_text()

        print("\nExample 2: Colors")
        example_colors()

        print("\nExample 3: Animations")
        example_animations()

        print("\nExample 4: Custom Frames")
        example_custom_frames()

        print("\nAll examples completed successfully!")

    except KeyboardInterrupt:
        print("\nExamples stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
