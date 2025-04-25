from lametric_stream.client import LMStream


with LMStream(
    api_key="7b36613762303235642d366534612d343831302d383138662d3332633566613636643235357d",
    ip="130.223.223.241",
) as stream:
    # Create a 2x2 grid of regions
    regions = stream.create_grid(rows=2, cols=2)

    # Top-left: Scrolling text
    stream.set_region_text(
        region_id=regions[0], text="Temperature: 72°F", text_color=(255, 255, 0)
    )

    # Top-right: Solid color
    stream.set_region_color(region_id=regions[1], color=(255, 0, 0))  # Red

    # Bottom-left: Gradient
    stream.set_region_color(
        region_id=regions[2], color=[(0, 255, 0), (0, 0, 255)]  # Green to blue
    )

    # Bottom-right: Another scrolling text
    stream.set_region_text(
        region_id=regions[3], text="Humidity: 45%", text_color=(255, 255, 255)
    )
