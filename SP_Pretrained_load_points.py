import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

base_dir = '/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/'

imgs_dir = os.path.join(base_dir, 'SCARF_tests/mustard_SCARF/frames')
points_dir = os.path.join(base_dir, 'SCARF_tests/mustard_SCARF/points/')
output_dir = os.path.join(base_dir, 'SCARF_tests/mustard_SCARF/points_comparison/SCARF')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith('.png')])
points_files = sorted([f for f in os.listdir(points_dir) if f.endswith('.npy')])

N_imgs = 186  # Number of files inside folder

for idx in range(N_imgs):
    
    # Load keypoints
    keypoints = np.load(os.path.join(points_dir, points_files[idx]))

    # Get x and y coordinates
    x = keypoints[0, :] * 12 # Check the img size
    y = keypoints[1, :] * 9  # Check the img size
    
    # 1920x1080 (frames extracted by SCARF) -> 1420x1080 without lateral bands -> 12*x, 9*y
    # 320x240   (images from SP) -> multiply dimension by 2
    # 640x480   (true size of dataset imgs) -> multiply by 4
    
    
    #print(x, y)

    img_dir = os.path.join(imgs_dir, image_files[idx])
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB, otherwise points will all be red

    # Plot img with points overlay
    plt.imshow(image)
    plt.scatter(x, y, color='red', s=10)
    
    # Save output
    output_path = os.path.join(output_dir, image_files[idx])
    plt.savefig(output_path)
    plt.close()

print(f"Images saved at {output_dir}")