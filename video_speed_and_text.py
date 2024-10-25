import os
import subprocess


input_video = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tuning/SP/alpha1_0-C0_1/alpha1_0-C0_1.mp4"
#slowed_down_video = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tuning/SP/alpha1_0-C0_1/slowed_down.mp4" 
precompression_video = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tuning/SP/alpha1_0-C0_1/precompression.mp4" 
output_video = "/home/cappe/Desktop/uni5/Tesi/IIT/Algorithms/SuperPointPretrainedNetwork-master/assets/SCARF_tuning/SP/alpha1_0-C0_1/alpha1_0-C0_1_github.mp4"   

# delete videos if they already exists

#if os.path.exists(slowed_down_video):
#  os.remove(slowed_down_video) 
  
if os.path.exists(precompression_video):
  os.remove(precompression_video) 
  
if os.path.exists(output_video):
  os.remove(output_video)   
  
  
'''
  
# ffmpeg -i video_2.mp4 -filter:v "setpts=8.0*PTS" test.mp4

# ffmpeg command to slow down video
command_speed = [
    "ffmpeg",
    "-i", input_video,
    "-filter:v", "setpts=4.0*PTS",  # slow down to 1/4
    slowed_down_video
]

subprocess.run(command_speed, check=True)

'''


# ffmpeg -i test.mp4 -vf drawtext=fontcolor=red:fontsize=30:text=1/8×speed:y=440 test2.mp4
# ffmpeg command to put text on video
command_text = [
    "ffmpeg",
    "-i", input_video,
    "-vf", "drawtext=fontcolor=yellow:fontsize=20:text='alpha=1.0 & C=0.1:x=10:y=20'",    # 1/4×speed, 
    precompression_video        #precompression_video/output_video
]

subprocess.run(command_text, check=True)

# ffmpeg -i input.mp4 -vcodec libx265 -crf 28 output.mp4
# ffmpeg -i input.mkv -vf "scale=trunc(iw/4)*2:trunc(ih/4)*2" half_the_frame_size.mkv # can reduce more?

# ffmpeg command to compress video
command_compress = [
    "ffmpeg",
    "-i", precompression_video,
    "-vcodec", "libx265", "-crf", "22",
    output_video
]

subprocess.run(command_compress, check=True)


print(f"Video elaborated and saved at {output_video}")

#if os.path.exists(input_video):
#  os.remove(input_video)
  
#if os.path.exists(slowed_down_video):
#  os.remove(slowed_down_video)
  
if os.path.exists(precompression_video):
  os.remove(precompression_video)