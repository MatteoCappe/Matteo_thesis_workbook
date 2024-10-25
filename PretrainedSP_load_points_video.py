import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

base_dir = '/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/'

# Paths to the video and points directory
video_path = os.path.join(base_dir, 'SCARF_tests/mustard_dumping/sensitivity50/events3/events3.mp4')
points_dir = os.path.join(base_dir, 'SCARF_tests/mustard_dumping/sensitivity50/events3/points/')
output_dir = os.path.join(base_dir, 'SCARF_tests/mustard_dumping/sensitivity50/events3/points_comparison')

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get list of points files
points_files = sorted([f for f in os.listdir(points_dir) if f.endswith('.npy')])

# Open the video
cap = cv2.VideoCapture(video_path)
N_frames = len(points_files)  # Number of points available
start_idx = 0  # Start processing from a specific frame if necessary

frame_idx = 1
while cap.isOpened() and frame_idx < N_frames:
    ret, frame = cap.read()  # Read frame from video
    
    if not ret:
        print("End of video or error reading the frame.")
        break
    
    if frame_idx >= start_idx:
        print(f"Processing frame {frame_idx}/{N_frames}")

        # Load corresponding keypoints
        keypoints = np.load(os.path.join(points_dir, points_files[frame_idx]))

        # Get x and y coordinates and scale them
        x = keypoints[0, :] * 4  # Scale by 4
        y = keypoints[1, :] * 4  # Scale by 4

        # Convert BGR to RGB (for matplotlib plotting)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Plot the frame with points overlay
        plt.imshow(frame_rgb)
        plt.scatter(x, y, color='red', s=10)

        # Save the output image
        output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
        plt.savefig(output_path)
        plt.close()

    frame_idx += 1

# Release the video capture object
cap.release()

print(f"Images with keypoints saved at {output_dir}")
