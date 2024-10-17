import subprocess
import cv2
import os

# Paths to your two videos
video1_path = '/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/test_RGB/mustard_RGB/SP/mustard_RGB_github_spedUP.mp4'      # change
video2_path = '/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tests/mustard_SCARF/SP/mustard_SCARF_github.mp4' # change

# Open the video files
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Get the properties of the videos (assume both videos have the same FPS)
fps = cap1.get(cv2.CAP_PROP_FPS)

# Get the width and height of the frames
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set the dimensions for both videos to be the same if they differ
new_width = min(width1, width2)
new_height = min(height1, height2)

# Create VideoWriter to save the output video
output_width = new_width * 2  # For side-by-side video, width is doubled
output_height = new_height
output_video = 'outputs/mustard_comparison.mp4' # change
output_video_github = 'outputs/mustard_comparison_github.mp4' # change

# delete video if it already exists
if os.path.exists(output_video):
  os.remove(output_video)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))

# Loop through the frames of both videos
while True:
    # Read frames from both videos
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # If either video ends, stop the loop
    if not ret1 or not ret2:
        break

    # Resize frames to the new dimensions (if required)
    frame1_resized = cv2.resize(frame1, (new_width, new_height))
    frame2_resized = cv2.resize(frame2, (new_width, new_height))

    # Concatenate frames horizontally
    combined_frame = cv2.hconcat([frame1_resized, frame2_resized])

    # Write the combined frame to the output video
    out.write(combined_frame)

    # Optionally show the frame
    cv2.imshow('Combined Video', combined_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()

if os.path.exists(output_video_github):
  os.remove(output_video_github)   

# ffmpeg command
command = [
    "ffmpeg",
    "-i", output_video,
    "-vcodec", "libx264",
    "-acodec", "aac",
    output_video_github
]

subprocess.run(command, check=True)

print(f"Video converted and saved at {output_video_github}")

if os.path.exists(output_video):
  os.remove(output_video)