import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
  scales = np.array(scales)
  aspect_ratios = np.array(aspect_ratios)
  stride = image_size // feature_size

  # Calculate center coordinates for each cell in the feature map
  center_x_coords = stride * (np.arange(feature_size) + 0.5)
  center_y_coords = stride * (np.arange(feature_size) + 0.5)

  # Generate all combinations of widths and heights from scales and aspect ratios
  # For each scale, apply all aspect ratios
  anchor_widths = np.array([s * np.sqrt(ar) for s in scales for ar in aspect_ratios])
  anchor_heights = np.array([s / np.sqrt(ar) for s in scales for ar in aspect_ratios])

  # Create a meshgrid of all center points
  grid_cx, grid_cy = np.meshgrid(center_x_coords, center_y_coords)
  grid_cx = grid_cx.flatten()
  grid_cy = grid_cy.flatten()

  # Expand center coordinates to match the number of anchor shapes
  # And expand anchor widths/heights to match the number of grid points
  final_cx = np.repeat(grid_cx, len(anchor_widths))
  final_cy = np.repeat(grid_cy, len(anchor_heights))
  final_widths = np.tile(anchor_widths, len(grid_cx))
  final_heights = np.tile(anchor_heights, len(grid_cy))

  # Calculate x_min, y_min, x_max, y_max for all anchors
  x_min = final_cx - final_widths / 2
  y_min = final_cy - final_heights / 2
  x_max = final_cx + final_widths / 2
  y_max = final_cy + final_heights / 2

  # Stack the coordinates to form the final anchors array
  anchors = np.stack([x_min, y_min, x_max, y_max], axis=1)

  return anchors.tolist()