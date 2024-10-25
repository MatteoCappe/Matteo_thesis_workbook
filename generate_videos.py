import subprocess
import cv2
import os

def generate_video_from_images(image_dir, output_video_path, fps=30):
    
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = sorted([img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])

    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    
    height, width, layers = frame.shape # Save values of first img

    # Init video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop through each image and add it to the video
    for image in images:
        img_path = os.path.join(image_dir, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release video
    video.release()
    print(f"Video saved at {output_video_path}")
    
# synthetic dataset: "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tuning/SP/alpha1_0-C0_1/alpha1_0-C0_1_points.mp4"

image_directory = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tests/mustard_dumping/sensitivity50/events5/SP"
output_video = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tests/mustard_dumping/sensitivity50/events5/SP_events5.mp4"
output_video_github = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tests/mustard_dumping/sensitivity50/events5/SP_event5_github.mp4"   

# delete video if it already exists
if os.path.exists(output_video):
  os.remove(output_video)

generate_video_from_images(image_directory, output_video, fps=240) # for dataset videos put 500, for real event camera video usually 120/240, for SP put the one of the original video?

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
