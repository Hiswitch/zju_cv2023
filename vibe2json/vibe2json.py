import joblib 
import numpy as np
import json

vibe_result = joblib.load('vibe_output.pkl')
vibe_result = vibe_result[1]

pred_cam = vibe_result['pred_cam']
orig_cam = vibe_result['orig_cam']
pose = vibe_result['pose']
betas = vibe_result['betas']
bboxes = vibe_result['bboxes']
frame_ids = vibe_result['frame_ids']

def get_camera_parameters(pred_cam, bbox):
    FOCAL_LENGTH_X = 851.39679195
    FOCAL_LENGTH_Y = 718.58171575
    CROP_SIZE = 224

    bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
    assert bbox_w == bbox_h

    bbox_size = bbox_w
    bbox_x = bbox_cx - bbox_w / 2.
    bbox_y = bbox_cy - bbox_h / 2.

    scale = bbox_size / CROP_SIZE

    cam_intrinsics = np.eye(3)
    cam_intrinsics[0, 0] = FOCAL_LENGTH_X * scale
    cam_intrinsics[1, 1] = FOCAL_LENGTH_Y * scale
    cam_intrinsics[0, 2] = bbox_size / 2. + bbox_x 
    cam_intrinsics[1, 2] = bbox_size / 2. + bbox_y

    cam_s, cam_tx, cam_ty = pred_cam
    trans = [cam_tx, cam_ty, (FOCAL_LENGTH_X + FOCAL_LENGTH_Y)/(CROP_SIZE*cam_s + 1e-9)]

    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = trans

    return cam_intrinsics, cam_extrinsics

cam_intrinsics = []
cam_extrinsics = []
for p, b in zip(pred_cam, bboxes):
    i, e = get_camera_parameters(p, b)
    cam_intrinsics.append(i)
    cam_extrinsics.append(e)

frames_data = {}
for frame in frame_ids:
    frame_data = {
        "poses": pose[frame].tolist(),
        "betas": betas[frame].tolist(),
        "cam_intrinsics": cam_intrinsics[frame].tolist(),
        "cam_extrinsics": cam_extrinsics[frame].tolist(),
    }

    frames_data[f'{frame:06d}'] = frame_data
    
json_data = json.dumps(frames_data, indent=2)

with open('metadata.json', 'w') as json_file:
    json_file.write(json_data)