import yaml
import numpy as np

# Load the cam_to_cam.yaml file
with open('interlaken_00_c/interlaken_00_c_calibration/cam_to_cam.yaml', 'r') as file:
    cam_to_cam = yaml.safe_load(file)

# Extract values for CAMERA_CALIBRATION_LEFT
cam0 = cam_to_cam['intrinsics']['cam0']
cam3 = cam_to_cam['intrinsics']['cam3']

# Extract extrinsic matrices
T_10 = np.array(cam_to_cam['extrinsics']['T_10'])
T_21 = np.array(cam_to_cam['extrinsics']['T_21'])
T_32 = np.array(cam_to_cam['extrinsics']['T_32'])

# Calculate HN as the multiplication of the three matrices
HN = T_10 @ T_21 @ T_32

# Create the output file content
output_content = f"""
[CAMERA_CALIBRATION_LEFT]

w {cam0['resolution'][0]}
h {cam0['resolution'][1]}
fx {cam0['camera_matrix'][0]}
fy {cam0['camera_matrix'][1]}
cx {cam0['camera_matrix'][2]}
cy {cam0['camera_matrix'][3]}
k1 {cam0['distortion_coeffs'][0]}
k2 {cam0['distortion_coeffs'][1]}
p1 {cam0['distortion_coeffs'][2]}
p2 {cam0['distortion_coeffs'][3]}
distortion_model b'{cam0['distortion_model']}'

[CAMERA_CALIBRATION_RIGHT]

w {cam3['resolution'][0]}
h {cam3['resolution'][1]}
fx {cam3['camera_matrix'][0]}
fy {cam3['camera_matrix'][1]}
cx {cam3['camera_matrix'][2]}
cy {cam3['camera_matrix'][3]}
k1 {cam3['distortion_coeffs'][0]}
k2 {cam3['distortion_coeffs'][1]}
p1 {cam3['distortion_coeffs'][2]}
p2 {cam3['distortion_coeffs'][3]}
distortion_model b'{cam3['distortion_model']}'

[STEREO_DISPARITY]
HN ({HN[0][0]} {HN[0][1]} {HN[0][2]} {HN[0][3]} {HN[1][0]} {HN[1][1]} {HN[1][2]} {HN[1][3]} {HN[2][0]} {HN[2][1]} {HN[2][2]} {HN[2][3]} 0. 0. 0. 1.)
"""

# Write the output content to a file
with open('interlaken_00_c/interlaken_00_c_calibration/camera_calibration.txt', 'w') as file:
    file.write(output_content)

print("File created successfully.")