import cv2
import cv2.aruco as aruco
import numpy as np

MARKER_SIZE = 200
MARGIN = 50

# Get dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Generate 4 marker images
marker_images = []
for marker_id in range(1, 5):
    marker_img = np.zeros((MARKER_SIZE, MARKER_SIZE), dtype=np.uint8)
    marker_img = aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE, marker_img, 1)
    marker_images.append(marker_img)

# Compute canvas size
canvas_width = 4*MARKER_SIZE
canvas_height = 4*MARKER_SIZE
canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255

# -----------------------------
# Perfect diamond formation
# -----------------------------
# Top row: Aruco 1
top_x = (canvas_width - MARKER_SIZE)//2
top_y = MARGIN

# Middle row: Aruco 2 (left) and Aruco 3 (right)
middle_y = MARKER_SIZE + 2*MARGIN
left_x = (canvas_width//2) - MARKER_SIZE - MARGIN//2
right_x = (canvas_width//2) + MARGIN//2

# Bottom row: Aruco 4 (centered below middle row)
bottom_x = (canvas_width - MARKER_SIZE)//2
bottom_y = middle_y + MARKER_SIZE + MARGIN

positions = [
    (top_x, top_y),        # Aruco 1
    (left_x, middle_y),    # Aruco 2
    (right_x, middle_y),   # Aruco 3
    (bottom_x, bottom_y)   # Aruco 4
]

# Place markers
for img, pos in zip(marker_images, positions):
    x, y = pos
    canvas[y:y+MARKER_SIZE, x:x+MARKER_SIZE] = img

# Show & save
cv2.imshow("Diamond ArUco", canvas)
cv2.imwrite("diamond_aruco_fixed.png", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Diamond ArUco image generated and saved as 'diamond_aruco_fixed.png'")
