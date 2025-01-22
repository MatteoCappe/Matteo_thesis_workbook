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
    disparity_map = disp_16bit.astype('float32')/256.0

    #print(disparity_map.shape)

    disparity_dict = {}
    for i in range(disparity_map.shape[0]):
        for j in range(disparity_map.shape[1]):
            if disparity_map[i, j] > 0:
                disparity_dict[f"({i},{j})"] = float(disparity_map[i, j])

    #with open(json_path, 'w') as json_file:
    #    json.dump(disparity_dict, json_file, indent=4)

    #print(f"Disparity map saved to {json_path}")

    
    return disparity_map


# Function to load keypoints and descriptors
def load_features(index):
    
    base_dir_left = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/DSEC_videos/SP/thun_00_a/1_5-0_3/rectified_timestamp_SCARF_left-1_5-0_3"
    base_dir_right = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/DSEC_videos/SP/thun_00_a/1_5-0_3/rectified_timestamp_SCARF_right-1_5-0_3"
        
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
    
# Visualization function
def visualize_disparity_map(disparity_map):
    # Create a blank color image
    color_disparity = np.zeros((disparity_map.shape[0], disparity_map.shape[1], 3), dtype=np.uint8)

    # Apply custom colormap
    color_disparity[disparity_map == -1] = [200, 200, 200]  # White for Ground Truth
    #color_disparity[disparity_map == -2] = [255, 165, 0]  # Blue for discarded matches
    #color_disparity[disparity_map > 0] = [0, 255, 0]  # Green for low values
    #color_disparity[disparity_map > disparity_threshold] = [0, 0, 255]  # Red for high values

    # Create masks for green and red dots
    green_mask = (disparity_map > 0) & (disparity_map <= disparity_threshold)
    red_mask = disparity_map > disparity_threshold
    blue_mask = disparity_map == -2

    # Create an overlay for larger dots
    overlay = np.zeros_like(color_disparity, dtype=np.uint8)

    # Define the radius and thickness for the dots
    dot_radius = 2
    dot_thickness = -1  # Filled circles

    # Draw larger green dots
    green_indices = np.argwhere(green_mask)
    for y, x in green_indices:
        cv2.circle(overlay, (x, y), dot_radius, (0, 255, 0), dot_thickness)

    # Draw larger red dots
    red_indices = np.argwhere(red_mask)
    for y, x in red_indices:
        cv2.circle(overlay, (x, y), dot_radius, (0, 0, 255), dot_thickness)
        
    # Draw larger blue dots
    blue_indices = np.argwhere(blue_mask)
    for y, x in blue_indices:
        top_left = (x - square_range, y - square_range)
        bottom_right = (x + square_range, y + square_range)
        cv2.rectangle(overlay, top_left, bottom_right, (255, 165, 0), dot_thickness)

    # Combine the overlay with the base color map
    color_disparity = cv2.addWeighted(color_disparity, 1.0, overlay, 0.7, 0)
    
    '''
    # Annotate each match with its unique identifier
    match_id = 0  # Start a counter for matches
    for i, match in enumerate(matches):
        if match != -1:  # Valid match
            xd0, yd0 = int(x0[i]), int(y0[i])  # Keypoint in left image
            xd1, yd1 = int(x1[match]), int(y1[match])  # Corresponding keypoint in right image
            
            # Draw identifier on the left image
            cv2.putText(
                color_disparity, str(match_id), (xd0, yd0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.3, (0, 255, 255), 1, cv2.LINE_AA
            )
            
            match_id += 1
    '''

    # Plot the combined disparity map
    height, width = disparity_map.shape
    dpi = 100  # Dots per inch for the figure
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill the entire figure with the image
    ax.imshow(color_disparity[..., ::-1])  # Convert BGR to RGB for Matplotlib
    ax.axis('off')  # Turn off axis

    # Save the output image
    output_path = os.path.join(maps_dir, f"{disparity_id:05}.png")
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    
    # like this i'm only plotting the keypoints, in order to have disparity map i need to plot the distances?

# Visualization function
def visualize_matches(stereo_left, stereo_right, img_left, img_right, disparity_id):
    img_left = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)
    
    # Define output paths for saving the images
    output_path_left = os.path.join(stereo_left_dir, f"{disparity_id:05}.png")
    output_path_right = os.path.join(stereo_right_dir, f"{disparity_id:05}.png")
    
    # Create masks for green, red and blue dots (mutually exclusive)
    green_mask_left = (stereo_left > 0) & (stereo_left <= disparity_threshold)
    red_mask_left = stereo_left > disparity_threshold    
    blue_mask_left = stereo_left == -2
    green_mask_right = (stereo_right > 0) & (stereo_right <= disparity_threshold)
    red_mask_right = stereo_right > disparity_threshold
    blue_mask_right = stereo_right== -2

    # Define the radius and thickness for the dots
    dot_radius = 2  # Adjust size as needed
    dot_thickness = -1  # Filled circles

    # Overlay green, red and blue dots on the left image
    for y, x in np.argwhere(green_mask_left):
        cv2.circle(img_left, (x, y), dot_radius, (0, 255, 0), dot_thickness)
    for y, x in np.argwhere(red_mask_left):
        cv2.circle(img_left, (x, y), dot_radius, (0, 0, 255), dot_thickness)
    for y, x in np.argwhere(blue_mask_left):
        cv2.circle(img_left, (x, y), dot_radius, (255, 0, 0), dot_thickness)

    # Overlay green, red and blue dots on the right image
    for y, x in np.argwhere(green_mask_right):
        cv2.circle(img_right, (x, y), dot_radius, (0, 255, 0), dot_thickness)
    for y, x in np.argwhere(red_mask_right):
        cv2.circle(img_right, (x, y), dot_radius, (0, 0, 255), dot_thickness)
    for y, x in np.argwhere(blue_mask_right):
        cv2.circle(img_right, (x, y), dot_radius, (255, 0, 0), dot_thickness)    
        
    '''
    # Annotate each match with its unique identifier
    match_id = 0  # Start a counter for matches
    for i, match in enumerate(matches):
        if match != -1:  # Valid match
            xd0, yd0 = int(x0[i]), int(y0[i])  # Keypoint in left image
            xd1, yd1 = int(x1[match]), int(y1[match])  # Corresponding keypoint in right image
            
            # Draw identifier on the left image
            cv2.putText(
                img_left, str(match_id), (xd0, yd0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.25, (0, 255, 255), 1, cv2.LINE_AA
            )
            
            # Draw identifier on the right image
            cv2.putText(
                img_right, str(match_id), (xd1, yd1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.25, (0, 255, 255), 1, cv2.LINE_AA
            )
            
            match_id += 1
    '''
    
    # Save the images with overlaid dots
    cv2.imwrite(output_path_left, img_left)
    cv2.imwrite(output_path_right, img_right)

disparity_id = 0
square_range = 2 # !!! # determine the range of pixels to consider for the disparity calculation (left, right, up, top: 1 -> 3x3)
disparity_threshold = 9 # 9, 6, 3 # !!!

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running inference on device \"{}\"'.format(device))
config = {
    'superglue': {
        'weights': 'indoor',
        'sinkhorn_iterations': 20,  # change value for sinkhorn iteration -> 90 should get higher accuracy -> try 50
        'match_threshold': 0.5,     # !!!change value for matching threshold!!!
    }
}
matching = Matching(config).eval().to(device)
keys = ['keypoints_left', 'keypoints_right', 'scores_left', 'scores_right', 'descriptors_left', 'descriptors_right']

output_dir = f"/home/cappe/Desktop/uni5/Tesi/IIT/EventPointDatasets/DSEC/evaluation_thun_00_a_rectified/1_5-0_3/{config['superglue']['match_threshold']}-{disparity_threshold}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# check VideoStreamer definition to check if its parameters are correct
video_left_path = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/DSEC_videos/SP/thun_00_a/1_5-0_3/rectified_timestamp_SCARF_left-1_5-0_3.mp4"
vs_left = VideoStreamer(video_left_path, resize=[640, 480], skip=1, image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=1000)     
video_right_path = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/DSEC_videos/SP/thun_00_a/1_5-0_3/rectified_timestamp_SCARF_right-1_5-0_3.mp4"
vs_right = VideoStreamer(video_right_path, resize=[640, 480], skip=1, image_glob=['*.png', '*.jpg', '*.jpeg'], max_length=1000)

frame_left, ret_left = vs_left.next_frame()
assert ret_left, 'Error when reading the first frame of the left video (check directory declaration?)'

frame_right, ret_right = vs_right.next_frame()
assert ret_right, 'Error when reading the first frame of the right video (check directory declaration?)'

matches_dir = os.path.join(output_dir, "matches")
if not os.path.exists(matches_dir):
    os.makedirs(matches_dir)
    
maps_dir = os.path.join(output_dir, "disparity_maps")
if not os.path.exists(maps_dir):
    os.makedirs(maps_dir)    

accuracy_txt_path = os.path.join(output_dir, "accuracy.txt")
with open(accuracy_txt_path, 'w') as f:
    f.write("---Accuracy---\n")
    
stereo_left_dir = os.path.join(output_dir, "stereo_left")
if not os.path.exists(stereo_left_dir):
    os.makedirs(stereo_left_dir)
    
stereo_right_dir = os.path.join(output_dir, "stereo_right")
if not os.path.exists(stereo_right_dir):
    os.makedirs(stereo_right_dir)

timer = AverageTimer()

avg_total_accuracy = 0
avg_total_ratio_num = 0
avg_total_ratio_den = 0
iterations = 0
zero_matches = 0

img_shape = (480, 640)

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
    
    stereo_left = np.zeros((480, 640), dtype=np.float32)
    stereo_right = np.zeros((480, 640), dtype=np.float32)
    
    disparity_map_GTT = np.copy(disparity_map_GT)    
    disparity_map_GTT[disparity_map_GTT > 0] = -1
    disparity_map_new = np.copy(disparity_map_GTT)
    
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

    #valid = matches > -1
    valid = np.zeros(max(kpts0.shape[0], kpts0.shape[0], kpts1.shape[0], kpts1.shape[0]), dtype=bool)

        
    # TODO: now make it work for the loop    

    #print("kpts0 shape: ", kpts0.shape)
    #print("kpts1 shape: ", kpts1.shape)

    #print("mkpts0 shape:", mkpts0.shape)
    #print("mkpts1 shape:", mkpts1.shape)

    #np.set_printoptions(threshold=np.inf)

    #print("matches shape:", matches.shape)
    #print("Element of matches: ", matches[matches != -1])
    
    #np.set_printoptions(threshold=1000)

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
    disparity_difference = 0
    disparity_dict = {}
    square_data_list = []
    
    #counter_matches = -1
    
    total_matches = np.count_nonzero(matches != -1)    # M refers to correct matches in event data pairs
    
    mkpts0 = []
    mkpts1 = []

    for i, match in enumerate(matches):
        if match != -1:
            # Get corresponding coordinates values for the two keypoints
            xd0 = int(x0[i])
            yd0 = int(y0[i])
            xd1 = int(x1[match])
            yd1 = int(y1[match])
            
            
            
            #print("xd0: ", xd0, "xd1: ", xd1)
            
            #print("Disparity match simplified y: ", abs(yd0-yd1))
            #print("Disparity match simplified x: ", abs(xd0-xd1))
            disparity_match = abs(np.sqrt((xd1 - xd0)**2 + (yd1 - yd0)**2))
            #print("Disparity match Eucliden: ", abs(np.sqrt((xd1 - xd0)**2 + (yd1 - yd0)**2)))   # TODO use euclidean distance?)   
            #if disparity_match < disparity_threshold:
            disparity_map_new[yd0, xd0] = disparity_match
                #disparity_matches += 1

            # Debug prints
            #print(f"disparity_map_GT({yd0},{xd0}) = {disparity_map_GT[yd0, xd0]}")
            #print(f"disparity_map_new({yd0},{xd0}) = {disparity_map_new[yd0, xd0]}")
            #print(f"Difference = {abs(disparity_map_GT[yd0, xd0] - disparity_map_new[yd0, xd0])}")
 
            disparity_dict[f"({yd0},{xd0})"] = disparity_match
            disparity_difference = abs((disparity_map_new[yd0, xd0]) - np.mean(disparity_map_GT[yd0-square_range:yd0+square_range, xd0-square_range:xd0+square_range]))
            
            '''
            square_region_list = disparity_map_GT[yd0-square_range:yd0+square_range, xd0-square_range:xd0+square_range].tolist()
            square_data_list.append({
                "center": (xd0, yd0),
                "data": square_region_list,
                "sequence": disparity_id
            })
            '''
            
            if disparity_map_new[yd0, xd0] > 0:
                #counter_matches += 1
                if disparity_difference <= disparity_threshold:
                    disparity_map_new[yd0, xd0] = disparity_difference # position determined by epipolar geometry?????? -> instead of [yd0, xd0]
                    stereo_left[yd0, xd0] = disparity_difference
                    stereo_right[yd1, xd1] = disparity_difference
                    disparity_matches += 1
                    
                    valid[match] = True
                    
                    mkpts0.append(kpts0[i])
                    mkpts1.append(kpts1[matches[i]])
                    
                    
                elif disparity_difference > disparity_threshold and np.mean(disparity_map_GT[yd0-square_range:yd0+square_range, xd0-square_range:xd0+square_range]) == 0:      # 
                    disparity_map_new[yd0, xd0] = -2 # position determined by epipolar geometry?????? -> instead of [yd0, xd0]
                    stereo_left[yd0, xd0] = -2
                    stereo_right[yd1, xd1] = -2
                    total_matches -= 1  # discarded -> so it gets removed out of the total possible matches to consider
                else:
                    stereo_left[yd0, xd0] = disparity_difference
                    stereo_right[yd1, xd1] = disparity_difference
                
            #print(counter_matches, ":", np.mean(disparity_map_GT[yd0-square_range:yd0+square_range, xd0-square_range:xd0+square_range]), "in", f"({xd0}, {yd0})")

    #with open("square_disparity_data.json", "w") as json_file:
    #    json.dump(square_data_list, json_file, indent=4) 
    
    #disparity_map = create_disparity_map_from_matches(x0, y0, x1, y1, matches, img_shape)
    visualize_disparity_map(disparity_map_new)
    visualize_matches(stereo_left, stereo_right, frame_left, frame_right, disparity_id)
    
    #sorted_disparity_dict = {k: disparity_dict[k] for k in sorted(disparity_dict.keys(), key=lambda x: (int(x.split(',')[0][1:]), int(x.split(',')[1][:-1])))}
    #with open('disparity_map_matches.json', 'w') as json_file:
    #    json.dump(disparity_dict, json_file, indent=4)                   

    print("Number of correct matches: ", disparity_matches)
    print("Total number of matches: ", total_matches)
    
    if total_matches == 0:
        accuracy = 0
        zero_matches += 1
    else:
        accuracy = disparity_matches / total_matches
        print("Accuracy: ", accuracy*100)  
        avg_total_ratio_num += disparity_matches
        avg_total_ratio_den += total_matches
        iterations += 1
                
    avg_total_accuracy += accuracy*100
    ratio = f"{disparity_matches}/{total_matches}"
    
    # Save accuracy values to a .txt file
    #with open(accuracy_txt_path, 'a') as f:
    #    f.write(f"Accuracy disparity_map_frame_{disparity_id}: {ratio} -> {100*accuracy:.4f}\n")
    
    light_green_transparent = np.array([144, 238, 144, 50]) / 255.0
    color = np.tile(light_green_transparent, (valid.sum(), 1))

    text = [
        ''
    ]
    k_thresh = 0.05                                             # CHECK
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Disparity Threshold: {:.2f}'.format(disparity_threshold),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Keypoints per image: {}:{}'.format(len(kpts0), len(kpts1)),
        #'Image Pair: {:05}:{:05}'.format(disparity_id, disparity_id),
    ]
    accuracy_text = [
        'Accuracy: {:.2f}%'.format(100 * accuracy),
        'Correct Matches: {}'.format(disparity_matches),
        'Total Matches: {}'.format(total_matches)
    ]
    out = make_matching_plot_fast(
        frame_left, frame_right, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=False, small_text=small_text, accuracy_text=accuracy_text)
    
    timer.update('viz')
    timer.print()

    if output_dir is not None:
        stem = 'matches_{:05}_{:05}'.format(disparity_id, disparity_id)
        out_file = str(Path(matches_dir, stem + '.png'))
        print('\nWriting image to {}'.format(out_file))
        cv2.imwrite(out_file, out)

    # Create the disparity map from the matches   
    disparity_id += 2
    
print(f"Average ratio: {avg_total_ratio_num}/{avg_total_ratio_den}\n")
print(f"Average accuracy: {(avg_total_accuracy/iterations):.4f}\n")
print("Threshold: ", disparity_threshold)
print(f"Number of sequences with no matches: {zero_matches}")
with open(accuracy_txt_path, 'a') as f:
    f.write(f"Average ratio: {avg_total_ratio_num}/{avg_total_ratio_den}\n")
    f.write(f"Average accuracy: {(avg_total_accuracy/iterations):.4f}\n")
    f.write(f"Threshold: {disparity_threshold}\n")
    f.write(f"Number of sequences with no matches: {zero_matches}")