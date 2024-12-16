import os
from pathlib import Path
import argparse
import time
import cv2
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching_pairs as Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

matplotlib.use('TkAgg')         # needed for plotting the disparity map, idk why
print(matplotlib.get_backend())

# disparity map created with rectified images

# Save the original map as dictionary to a JSON file
def get_disparity_matrix(disparity_map_path, json_path):
    disp_16bit = cv2.imread(disparity_map_path, cv2.IMREAD_ANYDEPTH)
    disparity_map = disp_16bit.astype('float32')/256

    #print(disparity_map.shape)

    '''
    disparity_dict = {}
    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            disparity_dict[f"({i},{j})"] = float(disparity_map[i, j])

    with open(json_path, 'w') as json_file:
        json.dump(disparity_dict, json_file, indent=4)

    print(f"Disparity map saved to {json_path}")
    '''
    
    return disparity_map

# Function to load keypoints and descriptors
def load_features(index):
    
    base_dir_left = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/DSEC_videos/SP/timestamps_thun_00_a_alpha1_3-C0_3-conf0_015_left"
    base_dir_right = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/DSEC_videos/SP/timestamps_thun_00_a_alpha1_3-C0_3-conf0_015_right"
        
    if not os.path.exists(base_dir_left) or not os.path.exists(base_dir_right):
        raise ValueError(f"Directory {base_dir_left} or {base_dir_right} do not exist, check the path")  
        
    if base_dir_left is not None:
        keypoints_path_left = base_dir_left + "/points/" + f"{index:05}.npy"
        scores_path_left = base_dir_left + "/scores/" + f"{index:05}.npy"
        descriptors_path_left = base_dir_left + "/descriptors/" + f"{index:05}.npy"
    
    if base_dir_right is not None:
        keypoints_path_right = base_dir_right + "/points/" + f"{index:05}.npy"
        scores_path_right = base_dir_right + "/scores/" + f"{index:05}.npy"
        descriptors_path_right = base_dir_right + "/descriptors/" + f"{index:05}.npy"
        
    if not os.path.exists (keypoints_path_left):
        
        # Calculate the average accuracy and then exit the program
        accuracy_txt_path = os.path.join(output_dir, "accuracy.txt")
        if os.path.exists(accuracy_txt_path):
            with open(accuracy_txt_path, 'r') as f:
                lines = f.readlines()
                accuracies = []
                for line in lines:
                    if 'Accuracy' in line:
                        try:
                            accuracies.append(float(line.split(': ')[1]))
                        except (IndexError, ValueError):
                            print(f"Skipping invalid line: {line.strip()}")
                if accuracies:
                    average_accuracy = sum(accuracies) / len(accuracies)    # N refers to the number of samples in a single sequence
                    print(f"Average Accuracy: {average_accuracy:.4f}")
                    print(f"Threshold: {disparity_threshold}")
                    
                    # Write the average accuracy at the end of the file
                    with open(accuracy_txt_path, 'a') as f_append:
                        f_append.write(f"Average Accuracy: {average_accuracy:.4f}\n")
                        f_append.write(f"Threshold: {disparity_threshold}\n")
                else:
                    print("No accuracy values found in accuracy.txt")
        else:
            print("accuracy.txt file not found")
        
        print(f"No more points found for index {index}. Stopping the program.")
        exit()
    
    keypoints_left = np.load(keypoints_path_left)
    scores_left = np.load(scores_path_left)
    descriptors_left = np.load(descriptors_path_left)
    
    keypoints_right = np.load(keypoints_path_right)
    scores_right = np.load(scores_path_right)
    descriptors_right = np.load(descriptors_path_right)
    
    # Convert the data to PyTorch tensors
    keypoints_left = torch.from_numpy(keypoints_left).float()
    scores_left = torch.from_numpy(scores_left).float()
    descriptors_left = torch.from_numpy(descriptors_left).float()
    
    keypoints_left = keypoints_left.transpose(0, 1)
    scores_left = scores_left.unsqueeze(0)
    
    # Convert the data to PyTorch tensors
    keypoints_right = torch.from_numpy(keypoints_right).float()
    scores_right = torch.from_numpy(scores_right).float()
    descriptors_right = torch.from_numpy(descriptors_right).float()
    
    keypoints_right = keypoints_right.transpose(0, 1)
    scores_right = scores_right.unsqueeze(0)
    
    #print("Keypoints shape:", keypoints.shape)  # Debugging: print the shape of keypoints
    #print("score shape:", scores.shape)  # Debugging: print the shape of keypoints
    #print("desc shape:", descriptors.shape)  # Debugging: print the shape of keypoints
    
    return {
        'keypoints_left': keypoints_left,
        'scores_left': scores_left,
        'descriptors_left': descriptors_left,
        'keypoints_right': keypoints_right,
        'scores_right': scores_right,
        'descriptors_right': descriptors_right
    }
    
### ?????????????????????????????????????????????????????????????????????????????????????????????????? ###    
        
# Function to create and visualize the disparity map
def create_disparity_map_from_matches(x0, y0, x1, y1, matches, img_shape):
    # Initialize a zero-filled disparity map with the same shape as the input image
    disparity_map = np.zeros(img_shape, dtype=np.float32)
    
    #disparity_map_GT[disparity_map_GT > 0] = -1 -> replace disparity_map with disparity_map_GT
        
    # Process each match
    for i, match in enumerate(matches):
        if match != -1:  # Only consider valid matches
            xd0 = int(x0[i])
            yd0 = int(y0[i])
            xd1 = int(x1[match])
            yd1 = int(y1[match])
                        
            disparity = abs(xd0 - xd1)
            
            # ???
            # incorrect calculation of disparity
            if xd0 < xd1:
                disparity_map[yd0, xd0:xd1] = disparity 
            else:
                disparity_map[yd0, xd1:xd0] = disparity
    
    return disparity_map

# Custom colormap function
def apply_custom_colormap(disparity_map):

    # Create a blank color image
    color_disparity = np.zeros((disparity_map.shape[0], disparity_map.shape[1], 3), dtype=np.uint8)

    # Apply custom colormap
    color_disparity[disparity_map < 0] = [200, 200, 200]  # White for Ground Truth
    color_disparity[disparity_map > 0] = [0, 255, 0]  # Green for low values
    color_disparity[disparity_map > disparity_threshold] = [0, 0, 255]  # Red for high values
    
    #TODO plot lines on top of the disparity_map_GT

    return color_disparity

# Visualization function
def visualize_disparity_map(disparity_map):
    # Apply custom colormap for better visualization
    color_disparity = apply_custom_colormap(disparity_map)

    # Ensure no resizing of frame
    height, width = disparity_map.shape
    dpi = 100  # Dots per inch for the figure
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Plot the disparity map
    ax = fig.add_axes([0, 0, 1, 1])  # Fill the entire figure with the image
    ax.imshow(color_disparity[..., ::-1])  # Convert BGR to RGB for Matplotlib

    ax.axis('off')  # Turn off axis

    # Save the output image
    output_path = os.path.join(maps_dir, f"{disparity_id:05}.png")
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    
    # like this i'm only plotting the keypoints, in order to have disparity map i need to plot the distances?
    
### ?????????????????????????????????????????????????????????????????????????????????????????????????? ###      

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,  # change value for sinkhorn iteration -> 90 should get higher accuracy
        'match_threshold': 0.2,     # !!!change value for matching threshold!!!
    }
}
matching = Matching(config).eval().to(device)
keys = ['keypoints_left', 'keypoints_right', 'scores_left', 'scores_right', 'descriptors_left', 'descriptors_right']

# check VideoStreamer definition to check if its parameters are correct
video_left_path = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/DSEC_videos/SP/thun_00_a_alpha1_3-C0_3_left/thun_00_a_alpha1_3-C0_3_left.mp4"
vs_left = VideoStreamer(video_left_path, resize=[640, 480], skip=1, image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=1000)     
video_right_path = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/DSEC_videos/SP/thun_00_a_alpha1_3-C0_3_right/thun_00_a_alpha1_3-C0_3_right.mp4"
vs_right = VideoStreamer(video_right_path, resize=[640, 480], skip=1, image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=1000)

frame_left, ret_left = vs_left.next_frame()
assert ret_left, 'Error when reading the first frame of the left video (check directory declaration?)'

frame_right, ret_right = vs_right.next_frame()
assert ret_right, 'Error when reading the first frame of the right video (check directory declaration?)'

output_dir = "/home/cappe/Desktop/uni5/Tesi/IIT/EventPointDatasets/DSEC/thun_00_a/evaluation/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

matches_dir = os.path.join(output_dir, "matches")
if not os.path.exists(matches_dir):
    os.makedirs(matches_dir)
    
maps_dir = os.path.join(output_dir, "disparity_maps")
if not os.path.exists(maps_dir):
    os.makedirs(maps_dir)    

accuracy_txt_path = os.path.join(output_dir, "accuracy.txt")
with open(accuracy_txt_path, 'w') as f:
    f.write("---Accuracy---\n")

disparity_id = 0
disparity_threshold = 9

timer = AverageTimer()

while True:
    
    frame_left, ret_left = vs_left.next_frame()
    frame_right, ret_right = vs_right.next_frame()
    
    if not ret_left:
        print('Finished demo_superglue.py')
        break
    timer.update('data')

    frame_tensor_left = frame2tensor(frame_left, device)
    current_data = load_features(vs_left.i - 1)  # Load features for image_id i
    current_data = {k+'1': current_data[k] for k in keys}
    current_data['image0'] = frame_tensor_left

    frame_tensor_right = frame2tensor(frame_right, device)
    #current_data_right = load_features(vs_right.i - 1)  # Load features for image_id i
    #current_data_right = {k+'1': current_data_right[k] for k in keys[1::2]}
    current_data['image1'] = frame_tensor_right

    input_data = {**current_data}
    
    # Deubg check
    if 'image0' not in input_data or 'image1' not in input_data:
        raise KeyError("Input data must contain 'image0' and 'image1' keys")

    pred = matching(input_data)
            
    #print("Pred keys:", pred.keys())
    
    # Load disparity map (ground truth)
    disparity_map_path = f"/home/cappe/Desktop/uni5/Tesi/IIT/EventPointDatasets/DSEC/thun_00_a/thun_00_a_disparity_event/{disparity_id:06}.png" # TODO iterate through all of them
    disparity_map_GT = get_disparity_matrix(disparity_map_path, json_path = "disparity_map.json")
    
    #np.set_printoptions(threshold=np.inf)
    #print("Disparity map GT: ", disparity_map_GT)

    kpts0 = current_data['keypoints_left1'].cpu().numpy()
    kpts1 = current_data['keypoints_right1'].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()
    timer.update('forward')
    
    # Debug check
    '''        
    print("Shape of kpts0:", kpts0.shape, "kpts0:", kpts0)
    print("Shape of kpts1:", kpts1.shape, "kpts1:", kpts1)
    print("Shape of matches:", matches.shape, "matches:", matches)
    print("Shape of confidence:", confidence.shape, "confidence:", confidence)
    '''

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]
    k_thresh = 0.05
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {:05}:{:05}'.format(disparity_id, disparity_id),
    ]
    out = make_matching_plot_fast(
        frame_left, frame_right, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=False, small_text=small_text)
    
    timer.update('viz')
    timer.print()

    if output_dir is not None:
        stem = 'matches_{:05}_{:05}'.format(disparity_id, disparity_id)
        out_file = str(Path(matches_dir, stem + '.png'))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)
        
    # TODO: now make it work for the loop    
    
    disparity_map_new = np.zeros((480, 640), dtype=np.float32)

    #print("kpts0 shape: ", kpts0.shape)
    #print("kpts1 shape: ", kpts1.shape)

    #print("mkpts0 shape:", mkpts0.shape)
    #print("mkpts1 shape:", mkpts1.shape)

    np.set_printoptions(threshold=np.inf)

    #print("matches shape:", matches.shape)
    print("Element of matches: ", matches[matches != -1])
    
    np.set_printoptions(threshold=1000)

    ## TODO
    # Create a dictionary to map keypoints to their matches
    # match_dict = {i: match for i, match in enumerate(matches) if match != -1}
    # convert to json file?

    # Fill the disparity map based on the matches

    # TODO iterate through all the timestamp-images and get the disparity map for each of them -> precision -> average

    x0 = kpts0[:, 0]
    y0 = kpts0[:, 1]
    x1 = kpts1[:, 0]
    y1 = kpts1[:, 1]

    #print("x0 shape: ", x0.shape)
    #print("y0 shape: ", y0.shape)

    # Get kpts0 coordinates
    # Get index of the match from matches
    # Get kpts1 coordinates
    # Calculate disparity as the difference between x0 and x1
    # Fill disparity map with disparity value -> color each pixel between the two points with the disparity value

    disparity_matches = 0

    for i, match in enumerate(matches):
        if match != -1:
            # Get corresponding coordinates values for the two keypoints
            xd0 = int(x0[i])
            yd0 = int(y0[i])
            xd1 = int(x1[match])
            yd1 = int(y1[match])
            
            print("xd0: ", xd0, "xd1: ", xd1)
            
            disparity_match = abs(xd0 - xd1)   # TODO use euclidean distance?  
            if disparity_match < disparity_threshold:
                disparity_map_new[yd0, xd0] = disparity_match
                disparity_matches += 1
            # TODO how to plot this?
                
    #for i in range(disparity_map_GT.shape[0]):
    #    for j in range(disparity_map_GT.shape[1]):
    #        if disparity_map_new[i, j] > 0 and abs(disparity_map_GT[i, j] - disparity_map_new[i, j]) < disparity_threshold:
    #            disparity_matches += 1
                
    total_matches = np.count_nonzero(matches != -1)    # M refers to correct matches in event data pairs       

    print("Number of correct matches: ", disparity_matches)
    print("Total number of matches: ", total_matches)

    if total_matches == 0:
        accuracy = 0
    else:
        accuracy = disparity_matches / total_matches
        print("Accuracy: ", accuracy*100)    
    
    # Save accuracy values to a .txt file
    with open(accuracy_txt_path, 'a') as f:
        f.write(f"Accuracy disparity_map_frame_{disparity_id}: {100*accuracy:.4f}\n")

    # Create the disparity map from the matches
    img_shape = (480, 640)
    disparity_map = create_disparity_map_from_matches(x0, y0, x1, y1, matches, img_shape)
    visualize_disparity_map(disparity_map)    
    disparity_id += 2
