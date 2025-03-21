# LaMetric Stream Library

A Python library for streaming custom content to LaMetric TIME or LaMetric SKY devices using the LMSP protocol.

## Overview

This library allows you to take full control of your LaMetric TIME or LaMetric SKY display by streaming custom pixel data over the local network. With this library, you can:

- Display text with custom fonts and character sets
- Show solid colors and gradients
- Create custom animations
- Stream real-time data visualizations

## Installation

```bash
pip install lametric-stream
```

## Quick Start

```python
from lametric_stream import LMStream, AnimationType

# Initialize with device API key and IP address
with LMStream(api_key="YOUR_API_KEY", ip="192.168.1.123") as stream:
    # Display text
    stream.send_text("Hello LaMetric!", text_color=(255, 255, 0))

    # Display a solid color for 3 seconds
    stream.send_solid_color((255, 0, 0), duration=3.0)

    # Send an animation
    stream.send_animation(
        animation_type=AnimationType.WAVE,
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        duration=5.0
    )
```

## Prerequisites

- A LaMetric TIME or LaMetric SKY device connected to your local network
- The device's IP address on your local network
- An API key for your device (found [here](https://developer.lametric.com/user/devices))

## Basic Usage

### Context Manager

The recommended way to use the library is with a context manager to ensure proper cleanup:

```python
with LMStream(api_key="YOUR_API_KEY", ip="192.168.1.123") as stream:
    # Your code here
```

### Text Display

Display text with full control over appearance:

```python
stream.send_text(
    text="Hello World!",
    text_color=(255, 255, 0),  # Yellow text
    bg_color=(0, 0, 0),        # Black background
    scroll=True,               # Scroll if text is too long
    scroll_speed=0.05,         # Speed of scrolling (seconds per column)
    initial_pause=1.0,         # Pause before scrolling starts
    end_pause=1.0,             # Pause at the end of scrolling
    clear_between_loops=True,  # Show blank screen between loops
    min_display_time=2.0       # Minimum display time for static text
)
```

### Colors and Gradients

```python
# Display a solid red for 3 seconds
stream.send_solid_color((255, 0, 0), duration=3.0)

# Display a horizontal gradient from red to blue for 3 seconds
stream.send_gradient(
    start_color=(255, 0, 0),
    end_color=(0, 0, 255),
    horizontal=True,
    duration=3.0
)
```

### Animations

```python
from lametric_stream import AnimationType

# Blink between red, green, and blue
stream.send_animation(
    animation_type=AnimationType.BLINK,
    colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
    duration=10.0
)

# Pulse animation with custom speed
stream.send_animation(
    animation_type=AnimationType.PULSE,
    colors=[(255, 0, 0), (0, 0, 255)],
    speed=2.0  # 2x normal speed
)

# Wave animation
stream.send_animation(
    animation_type=AnimationType.WAVE,
    colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
    speed=1.0
)
```

### Custom Frames

For advanced use cases, you can send custom pixel data:

```python
# Create a custom frame with a pattern
width, height = stream.canvas_size.get("width"), stream.canvas_size.get("height")
custom_frame = []

for y in range(height):
    for x in range(width):
        if (x + y) % 2 == 0:
            custom_frame.append((255, 0, 0))  # Red
        else:
            custom_frame.append((0, 0, 255))  # Blue

# Send the custom frame
stream.send_frame(pixels=custom_frame)
```

## Text Rendering Features

The library includes a robust text rendering system with the following features:

### Character Support

The built-in font supports:

- Uppercase and lowercase letters
- Numbers
- Special characters and punctuation
- Characters with descenders (g, j, p, q, y)

### Text Animation

- Static display for text that fits on screen
- Automatic scrolling for text that exceeds screen width
- Configurable scrolling speed
- Initial pause before scrolling starts
- End pause with clear screen between loops
- Minimum display time for short, static text

### Text Customization

- Custom text color
- Custom background color
- Vertical centering

## Animation Types

The library supports several built-in animations through the `AnimationType` enum:

- `AnimationType.BLINK`: Alternates between specified colors
- `AnimationType.PULSE`: Fades smoothly between specified colors
- `AnimationType.WAVE`: Creates a wave-like pattern moving across the display

## Advanced Usage

### Custom Animation Sequences

```python
# Create multiple frames
frames = [
    [(255, 0, 0)] * (width * height),  # Red frame
    [(0, 255, 0)] * (width * height),  # Green frame
    [(0, 0, 255)] * (width * height)   # Blue frame
]

# Specify custom durations for each frame
frame_durations = [0.5, 1.0, 0.5]  # In seconds

# Send the frames as an animation
stream.send_frames(
    frames=frames,
    frame_durations=frame_durations,
    loop=True,
    loop_count=3  # Loop 3 times (0 for infinite)
)
```

### Low-Level Control

For maximum control, you can work directly with the LMSP protocol:

```python
from lametric_stream import CanvasArea, ContentEncoding

# Create and send a custom LMSP packet
area = CanvasArea(
    x=0,
    y=0,
    width=canvas_width,
    height=canvas_height,
    pixels=my_custom_pixels
)
packet = stream._create_lmsp_packet(area, encoding=ContentEncoding.RAW)
stream._send_udp_packet(packet)
```

## API Reference

### Main Class

- `LMStream(api_key, ip, max_retries=3)`: Initialize the LaMetric streaming client

### Common Methods

- `send_text(text, text_color, bg_color, ...)`: Display text with various options
- `send_solid_color(color, duration)`: Display a solid color
- `send_gradient(start_color, end_color, horizontal, duration)`: Display a gradient
- `send_animation(animation_type, colors, duration, speed)`: Display built-in animations

### Enums

- `AnimationType`: Enum for animation types (BLINK, PULSE, WAVE)
- `ContentEncoding`: Enum for content encoding types (RAW, PNG, JPEG, GIF)
- `RenderMode`: Enum for render modes (PIXEL, TRIANGLE)
- `FillType`: Enum for fill types (SCALE, TILE)
- `PostProcessType`: Enum for post-processing types (NONE, EFFECT)

### Advanced Methods

- `send_frame(pixels, ...)`: Send a single frame of pixel data
- `send_frame_for_duration(pixels, duration, ...)`: Send a frame for a specific duration
- `send_frames(frames, fps, frame_durations, ...)`: Send a sequence of frames with custom timing

## Error Handling

The library includes custom exceptions for better error handling:

```python
from lametric_stream import LMStreamError, APIError, ProtocolError

try:
    with LMStream(api_key="YOUR_API_KEY", ip="192.168.1.123") as stream:
        stream.send_text("Hello LaMetric!")
except APIError as e:
    print(f"API error: {e}")
except ProtocolError as e:
    print(f"Protocol error: {e}")
except LMStreamError as e:
    print(f"General streaming error: {e}")
```

## Notes and Limitations

- The API key and IP address must be correct for successful connection
- Maximum framerate is limited by the device's capabilities
- The canvas size depends on your specific LaMetric model
- For optimal performance, keep animations simple on devices with limited resources

## Library Structure

```
lametric_stream/
├── __init__.py       # Exports main classes and enums
├── exceptions.py     # Custom exception classes
├── fonts.py          # Font definitions
├── client.py         # Main LMStream client
└── utils.py          # Utility functions
```

## Example Applications

### Weather Display

```python
# Display current temperature
def show_temperature(temp, unit="C"):
    with LMStream(api_key=API_KEY, ip=IP_ADDRESS) as stream:
        stream.send_text(f"{temp}°{unit}", text_color=(255, 255, 0))
```

### Status Monitor

```python
# Display system status with color coding
def show_status(status):
    with LMStream(api_key=API_KEY, ip=IP_ADDRESS) as stream:
        if status == "ok":
            stream.send_text("OK", text_color=(0, 255, 0))
        elif status == "warning":
            stream.send_text("WARNING", text_color=(255, 255, 0))
        else:
            stream.send_text("ERROR", text_color=(255, 0, 0))
```

## License

This library is released under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
