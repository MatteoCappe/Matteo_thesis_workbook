import cv2
import os

def generate_video_from_images(image_dir, output_video_path, fps=30):
    # Get list of images in the directory
    images = sorted([img for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])

    # Read the first image to get the width and height
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Loop through each image and add it to the video
    for image in images:
        img_path = os.path.join(image_dir, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved at {output_video_path}")

# Example usage
image_directory = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/test_SCARF"
output_video = "output_video.avi"
generate_video_from_images(image_directory, output_video, fps=30)
