import open3d as o3d
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import imageio

# Read OBJ file
input_folder_path = '../dance_models'
img_num = 0

# Set video properties (you can adjust these as needed)
fps = 30
output_video_path = 'output_video.mp4'

# Create a VideoWriter object using imageio
kwargs = { 'macro_block_size': None }
video_writer = imageio.get_writer(output_video_path, fps=fps, **kwargs)

for filename in tqdm(sorted(os.listdir(input_folder_path))):
    if filename.endswith('.obj'):
        input_file_path = os.path.join(input_folder_path, filename)

        mesh = o3d.io.read_triangle_mesh(input_file_path)

        # Read texture image using Pillow
        texture = Image.open("../videoavatars_output/tex-yang.jpg")

        # Perform vertical flip using Pillow and convert to NumPy array
        flipped_texture_data = np.array(texture.transpose(Image.FLIP_TOP_BOTTOM))

        # Create Open3D Image from NumPy array
        flipped_texture = o3d.geometry.Image(flipped_texture_data)

        # Assign the flipped texture to the mesh
        mesh.textures = [flipped_texture]

        # Create Mesh object
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

        # Create visualization window and add the model
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh)
        rendered_image = vis.capture_screen_float_buffer(True)  # True for depth
        # Convert the rendered image to a numpy array
        image_array = np.asarray(rendered_image)

        # Append the image array to the video
        video_writer.append_data((image_array * 255).astype(np.uint8))

# Close the VideoWriter object
video_writer.close()

print(f"Video created: {output_video_path}")
