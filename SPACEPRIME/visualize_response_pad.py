import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import degrees_va_to_pixels
from SPACEPRIME import get_data_path
import os

# ===================================================================
# Configuration (Must match time_resolved_dwell_time.py)
# ===================================================================
WIDTH = 1920
HEIGHT = 1080
DG_VA = 1
SCREEN_SIZE_CM_Y = 30
SCREEN_SIZE_CM_X = 53
VIEWING_DISTANCE_CM = 70
ROI_WIDTH = 200
ROI_HEIGHT = 200

def main():
    # Create a figure with 1:1 pixel mapping (approximate based on DPI)
    # We set a large figsize and DPI to ensure we have enough pixels
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]) # Full screen axes
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_axis_off()
    
    # Draw screen border
    ax.add_patch(patches.Rectangle((0, 0), WIDTH, HEIGHT, fill=False, edgecolor='black', linewidth=2))

    center_x = WIDTH / 2
    center_y = HEIGHT / 2
    
    # Draw Center
    ax.plot(center_x, center_y, 'r+', markersize=20, markeredgewidth=2, label='Center')

    # Calculate pixel offsets for the digits (DG_VA = 2 degrees)
    offset_x = degrees_va_to_pixels(
        degrees=DG_VA, screen_pixels=WIDTH, screen_size_cm=SCREEN_SIZE_CM_X, viewing_distance_cm=VIEWING_DISTANCE_CM
    )
    offset_y = degrees_va_to_pixels(
        degrees=DG_VA, screen_pixels=HEIGHT, screen_size_cm=SCREEN_SIZE_CM_Y, viewing_distance_cm=VIEWING_DISTANCE_CM
    )

    # Define positions relative to center (Numpad layout logic from main script)
    # Note: If origin is bottom-left, +y is Up.
    digit_positions = {
        7: (-offset_x, -offset_y), 8: (0, -offset_y), 9: (offset_x, -offset_y),
        4: (-offset_x, 0),         5: (0, 0),         6: (offset_x, 0),
        1: (-offset_x, offset_y),  2: (0, offset_y),  3: (offset_x, offset_y)
    }

    print(f"Screen Center: ({center_x}, {center_y})")
    print(f"Digit Offset X (pixels for {DG_VA} deg): {offset_x:.2f}")
    print(f"Digit Offset Y (pixels for {DG_VA} deg): {offset_y:.2f}")

    for digit, (dx, dy) in digit_positions.items():
        x = center_x + dx
        y = center_y + dy
        ax.text(x, y, str(digit),
                color='black', alpha=0.8, ha='center', va='center',
                fontsize=10, fontweight='bold')
        # Draw a small circle at the digit location
        ax.add_patch(patches.Circle((x, y), radius=10, color='blue', alpha=0.3))

    # Draw the ROI box used in the plots
    roi_x = (WIDTH - ROI_WIDTH) / 2
    roi_y = (HEIGHT - ROI_HEIGHT) / 2
    
    ax.add_patch(patches.Rectangle((roi_x, roi_y), ROI_WIDTH, ROI_HEIGHT, 
                                   fill=False, edgecolor='green', linewidth=3, linestyle='--', label='Zoom ROI'))
    
    # Save
    output_path = os.path.join(get_data_path(), 'plots', 'response_pad_full_screen.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved full screen response pad image to: {output_path}")

if __name__ == "__main__":
    main()