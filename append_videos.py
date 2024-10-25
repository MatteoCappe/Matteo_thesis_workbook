import os
import subprocess
import cv2

def write_video(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

base_dir = "/home/cappe/Desktop/uni5/Tesi/IIT/code/dataset/mustard_bottle_sequence1/SCARF_tuning/testing"

video1 = os.path.join(base_dir, 'events1.mp4')
video2 = os.path.join(base_dir, 'events2.mp4')
video3 = os.path.join(base_dir, 'events3.mp4')

output = os.path.join(base_dir, 'events.mp4')
output_github = os.path.join(base_dir, 'alpha1_0-C0_1.mp4')

cap1 = cv2.VideoCapture(video1)
cap2 = cv2.VideoCapture(video2)
cap3 = cv2.VideoCapture(video3)

fps = int(cap1.get(cv2.CAP_PROP_FPS))
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output, fourcc, fps, (width, height))

write_video(cap1)
write_video(cap2)
write_video(cap3)

out.release()

print(f"Concatenated video saved at {output}")

if os.path.exists(output_github):
  os.remove(output_github)   

# ffmpeg command
command = [
    "ffmpeg",
    "-i", output,
    "-vcodec", "libx264",
    "-acodec", "aac",
    output_github
]

subprocess.run(command, check=True)

print(f"Video converted and saved at {output_github}")

if os.path.exists(output):
  os.remove(output)
