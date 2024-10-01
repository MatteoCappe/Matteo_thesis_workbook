import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

base_path = '/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SP_Pytorch/data/synthetic_shapes/draw_checkerboard/'

img_path = os.path.join(base_path, 'images/test/234.png')
points_path = os.path.join(base_path, 'points/test/234.npy')

# Load data
img = cv2.imread(img_path)
point_array = np.load(points_path)
overlay = img.copy()

# Loop through the points in point_array and draw them
for point in point_array:
    x, y = int(point[0]), int(point[1])
    cv2.circle(overlay, (x, y), radius=3, color=(0, 255, 0), thickness=1)  # Draw green circles around point

# Create a window
cv2.namedWindow('ImageWindow', cv2.WINDOW_AUTOSIZE)

# Display the image with the drawn points
cv2.imshow('ImageWindow', overlay)
cv2.waitKey(0)  # Close window or stop execution

# Close the window
cv2.destroyAllWindows()
