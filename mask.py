import cv2
import os

img_dir = '/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/test_RGB/mustard_RGB/SP'
mask_dir = '/home/cappe/Desktop/uni5/Tesi/IIT/code/dataset/mustard_bottle_sequence1/MASK_imgs'
output_dir = '/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/test_RGB/RGB/masked_SP'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images = sorted([img for img in os.listdir(img_dir) if img.endswith(('.png', '.jpg', '.jpeg'))])
masks = sorted([img for img in os.listdir(mask_dir) if img.endswith(('.png', '.cs.png', '.jpg', 'jpeg'))])

i = 1

while(i < 3100):
    img_path = os.path.join(img_dir, f'frame_{i:05d}.png')
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=2, fy=2) # fx and fy multiplication factors for img size
    
    mask_path = os.path.join(mask_dir, f'{i:06d}.cs.png')
    mask = cv2.imread(mask_path, 0)
    
    res = cv2.bitwise_and(img, img, mask=mask)
    
    output_name = os.path.join(output_dir, f'frame_{i:05d}.png')
    cv2.imwrite(output_name, res)
    
    i+=1
    
print("done!")