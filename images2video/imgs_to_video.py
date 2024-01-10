import imageio
import os
from tqdm import tqdm

# Directory containing your image files
image_dir = '../dance_imgs'

# List all image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Sort the image files to maintain order in the video
image_files.sort()

# Define the output video file
output_video_path = 'output_video.mp4'

# Set video properties (you can adjust these as needed)
fps = 30

# Create a VideoWriter object using imageio
video_writer = imageio.get_writer(output_video_path, fps=fps)

# Iterate through each image file and add it to the video
for image_file in tqdm(image_files):
    image_path = os.path.join(image_dir, image_file)
    frame = imageio.imread(image_path)
    video_writer.append_data(frame)

# Close the VideoWriter object
video_writer.close()

print(f"Video created: {output_video_path}")
