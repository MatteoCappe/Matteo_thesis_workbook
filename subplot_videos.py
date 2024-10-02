import cv2

# Paths to your two videos
video1_path = 'outputs/SP_RGB_save_SpedUP.mp4'
video2_path = 'outputs/SP_SCARF_save.mp4'

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
output_video_path = 'outputs/SP_save_comparison.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

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
