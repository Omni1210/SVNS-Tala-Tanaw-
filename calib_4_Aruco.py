import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os

def generate_diamond_aruco(
    ids=[0, 1, 2, 3],
    marker_size_px=200,
    centroid_spacing_px=400,
    margin_px=150,
    output_path=r"D:\Users\Admin\Downloads\ERC\Aruco_Markers\diamond_aruco_4x4_50.png"
):
    """
    Generate 4 ArUco markers in diamond layout and save to the specified path.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # Create individual markers
    markers = [aruco.generateImageMarker(dictionary, id, marker_size_px) for id in ids]

    m = marker_size_px

    # Canvas size
    inner_width = 2 * m + centroid_spacing_px
    inner_height = 2 * m + centroid_spacing_px
    canvas_width = inner_width + 2 * margin_px
    canvas_height = inner_height + 2 * margin_px

    canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255  # white

    cx = canvas_width // 2
    cy = canvas_height // 2

    positions = [
        (cx, cy - centroid_spacing_px // 2),          # North
        (cx - centroid_spacing_px // 2, cy),          # West
        (cx + centroid_spacing_px // 2, cy),          # East
        (cx, cy + centroid_spacing_px // 2),          # South
    ]

    for marker_img, (x_center, y_center) in zip(markers, positions):
        half = m // 2
        x_start = x_center - half
        y_start = y_center - half

        if x_start < 0 or y_start < 0 or x_start + m > canvas_width or y_start + m > canvas_height:
            print(f"Warning: Marker at ({x_center}, {y_center}) is out of bounds.")
            continue

        canvas[y_start:y_start + m, x_start:x_start + m] = marker_img

    # Save the image
    success = cv2.imwrite(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if success:
        print(f"Successfully saved to: {output_path}")
        print(f"  Marker size: {m} px")
        print(f"  Centroid-to-centroid spacing: {centroid_spacing_px} px")
        print(f"  Total image size: {canvas_width} × {canvas_height} px")
        print(f"  Margin: {margin_px} px")
    else:
        print(f"Failed to save image to {output_path} — check folder permissions or path.")

    # Show preview (optional)
    cv2.imshow("Diamond ArUco 4x4_50", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 4 ArUco markers in diamond layout")
    parser.add_argument("--ids", type=int, nargs=4, default=[0,1,2,3],
                        help="Four marker IDs (default: 0 1 2 3)")
    parser.add_argument("--marker-size", type=int, default=200,
                        help="Size of each marker in pixels")
    parser.add_argument("--spacing", type=int, default=400,
                        help="Centroid-to-centroid spacing in pixels")
    parser.add_argument("--margin", type=int, default=150,
                        help="Margin around pattern")
    parser.add_argument("--output", type=str,
                        default=r"D:\Users\Admin\Downloads\ERC\Aruco_Markers\diamond_aruco_4x4_50.png",
                        help="Full output path")

    args = parser.parse_args()

    generate_diamond_aruco(
        ids=args.ids,
        marker_size_px=args.marker_size,
        centroid_spacing_px=args.spacing,
        margin_px=args.margin,
        output_path=args.output
    )