import joblib 
import numpy as np
import json

vibe_result = joblib.load('../dance/vibe_output.pkl')

vibe_result = vibe_result[1]
poses = vibe_result['pose']

for i in range(len(poses)):
    pose = poses[i].reshape((24, 3))
    np.save(f'../dance_poses/pose_{i:04d}.npy', pose)
