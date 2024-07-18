import os
import json
import cv2  
#from PIL import Image #imshow()

base_dir = 'dataset\\photorealistic1'

### Function to generate the video divided into 3 subplots to show the different types of images ###
def generate_video():
    video_name = 'first_generated_video.avi'
    
    frame_rgb = cv2.imread(os.path.join(base_dir, '000000.png'))
    height, width, layers = frame_rgb.shape
    
    video = cv2.VideoWriter(video_name, 0, 1, (width * 3, height))  # Width is tripled for 3 subplots
    
    # Define the base directory and file name format
    start_index = 0
    end_index = 1220  # Change this to the maximum number of images

    # Loop through each file index
    for i in range(start_index, end_index + 1):
        # Format the file name with leading zeros    
        file_name_rgb = f'{i:06}.png'  # :06 ensures 6 digits with leading zeros
        file_path_rgb = os.path.join(base_dir, file_name_rgb)
        
        # Check if the file exists (not really needed but to avoid possible errors)
        if not os.path.isfile(file_path_rgb):
            print(f"File not found: {file_name_rgb}")
            continue
        
        # Format the file name with leading zeros   
        file_name_event = f'{i:06}._ec.png'  # :06 ensures 6 digits with leading zeros
        file_path_event = os.path.join(base_dir, file_name_event)
        
        # Check if the file exists (not really needed but to avoid possible errors)
        if not os.path.isfile(file_path_event):
            print(f"File not found: {file_name_event}")
            continue
        
        # Format the file name with leading zeros   
        file_name_depth = f'{i:06}.depth.mm.16.png'  # :06 ensures 6 digits with leading zeros
        file_path_depth = os.path.join(base_dir, file_name_depth)
        
        # Check if the file exists (not really needed but to avoid possible errors)
        if not os.path.isfile(file_path_depth):
            print(f"File not found: {file_name_depth}")
            continue
        
        # Read images
        img_rgb = cv2.imread(file_path_rgb)
        img_event = cv2.imread(file_path_event)
        img_depth = cv2.imread(file_path_depth)
        
        # Resize images if they are not the same size as the first one
        if img_rgb.shape != (height, width, layers):
            img_rgb = cv2.resize(img_rgb, (width, height))
        if img_event.shape != (height, width, layers):
            img_event = cv2.resize(img_event, (width, height))
        if img_depth.shape != (height, width, layers):
            img_depth = cv2.resize(img_depth, (width, height))
        
        # Concatenate images horizontally for subplots
        images_combined = cv2.hconcat([img_rgb, img_event, img_depth])
        
        # Write frame to video
        video.write(images_combined)
        
    # Release the video
    video.release()
    
#Note: the video is 20 minutes long, so it's going to take a while to compile



### Process Camera settings ###
class Camera:
    # Initialize the object camera with its intrinsic parameters
    def __init__(self, resX, resY, fx, fy, cx, cy, s): 
        self.resX = resX  # Resolution in X direction
        self.resY = resY  # Resolution in Y direction
        self.fx = fx      # Focal length in X direction
        self.fy = fy      # Focal length in Y direction
        self.cx = cx      # Center X coordinate
        self.cy = cy      # Center Y coordinate
        self.s = s        # ?, ask what s is
        
    # Print the data from the intrinsic parameters of the camera
    def printData(self):    
        print(f"Resolution X: {self.resX}")
        print(f"Resolution Y: {self.resY}")
        print(f"Focal Length X (fx): {self.fx}")
        print(f"Focal Length Y (fy): {self.fy}")
        print(f"Principal Point X (cx): {self.cx}")
        print(f"Principal Point Y (cy): {self.cy}")
        print(f"S: {self.s}")
        print('\n')

try:
    with open(os.path.join(base_dir, '_camera_settings.json')) as file:        
        # Load data from json file
        data = json.load(file)
        
        # Read from intrnsic param and initialize object
        params = data['camera_settings'][0]['intrinsic_settings']
        camera = Camera(params['resX'], params['resY'], params['fx'], params['fy'], params['cx'], params['cy'], params['s'])
        
        # Print the intrinsic parameters
        camera.printData()
               
except FileNotFoundError:
    print("The camera_settings file was not found")
    
    
    
### Process Object data and load images ###
class Obj:
    # Initialize the object camera with its parameters
    def __init__(self, location_worldframe, quaternion_xyzw_worldframe, location, quaternion_xyzw, pose_transform, timestamp):
        self.location_worldframe = location_worldframe                  # Location fo the camera in the world frame
        self.quaternion_xyzw_worldframe = quaternion_xyzw_worldframe    # Quaternion orientation of the camera in the world frame
        self.location = location                                        # Location of the object in the camera frame
        self.quaternion_xyzw = quaternion_xyzw                          # Quaternion orientation of the object in the camera frame
        self.pose_transform = pose_transform                            # Full pose of the object wrt the camera frame?
        self.timestamp = timestamp                                      # Timestamp of when the picture is taken
        
    # Print the data from the parameters of the object
    def printData(self):
        print(f"Location World Frame: {self.location_worldframe}")
        print(f"Quaternion XYZW World Frame: {self.quaternion_xyzw_worldframe}")
        print(f"Location In Camera Frame: {self.location}")
        print(f"Quaternion XYZW Camera Frame: {self.quaternion_xyzw}")
        print(f"Pose Transform: {self.pose_transform}")
        print(f"Timestamp: {self.timestamp}")
        print('\n')
            
# Define the base directory and file name format
start_index = 0
end_index = 1220  # Change this to the maximum number of images

#TODO: create a function to check for the existance of the files

# Loop through each file index
for i in range(start_index, end_index + 1):
    # Format the file name with leading zeros
    file_name = f'{i:06}{".json"}'  # :06 ensures 6 digits with leading zeros
    file_path = os.path.join(base_dir, file_name)
    
    # Check if the file exists (not really needed but to avoid possible errors)
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        continue
    
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
        camera_data = data['camera_data']
        obj_data = data['objects'][0]
        
        obj = Obj(camera_data['location_worldframe'], camera_data['quaternion_xyzw_worldframe'], obj_data['location'],
                  obj_data['quaternion_xyzw'], obj_data['pose_transform'], data['timestamp'])
        
    # Process the data
    print(f"Processing file: {file_name}")
    #obj.printData()    # Uncomment to see the parameters of every object.json file, Warning: it's a lot of data
    
    
    
### Create a video with the images divided in subplots ###
generate_video()  