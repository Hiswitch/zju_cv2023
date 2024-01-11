import argparse
import os
import sys
from tqdm import tqdm

import os
os.environ['DISPLAY'] = ':1'

import numpy as np
from scipy.spatial.transform import Rotation

# to make run from console for module import
sys.path.append(os.path.abspath('..'))

from main.config import Config
from main.model import Model
from visualise.trimesh_renderer import TrimeshRenderer
from visualise.vis_util import preprocess_image, visualize


def rotmat2rotvec(pose):
    num_joints = pose.shape[0]
    axis_angle_params = np.zeros((num_joints, 3))

    for i in range(num_joints):
        rotation_matrix = pose[i]
        rotation = Rotation.from_matrix(rotation_matrix)
        axis_angle_params[i] = rotation.as_rotvec()
    
    return axis_angle_params

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo HMR2.0')

    # parser.add_argument('--image', required=False, default='frame_0000.png')
    parser.add_argument('--model', required=False, default='base_model', help="model from logs folder")
    parser.add_argument('--setting', required=False, default='paired', help="setting of the model")
    parser.add_argument('--joint_type', required=False, default='cocoplus', help="<cocoplus|custom>")
    parser.add_argument('--init_toes', required=False, default=False, type=str2bool,
                        help="only set to True when joint_type=cocoplus")

    args = parser.parse_args()
    if args.init_toes:
        assert args.joint_type, "Only init toes when joint type is cocoplus!"


    class DemoConfig(Config):
        BATCH_SIZE = 1
        ENCODER_ONLY = True
        LOG_DIR = os.path.abspath('../../logs/{}/{}'.format(args.setting, args.model))
        INITIALIZE_CUSTOM_REGRESSOR = args.init_toes
        JOINT_TYPE = args.joint_type

    config = DemoConfig()

    # initialize model
    model = Model()
    pose_num = 0

    input_folder_path = '../../../dance_frames/'
    # original_img, input_img, params = preprocess_image('images/{}'.format(args.image), config.ENCODER_INPUT_SHAPE[0])

    for filename in tqdm(sorted(os.listdir(input_folder_path))):
        if filename.endswith(".png"):
            original_img, input_img, params = preprocess_image(os.path.join(input_folder_path, filename), config.ENCODER_INPUT_SHAPE[0])
            result = model.detect(input_img)
            pose = np.squeeze(result['pose'].numpy())
            pose = rotmat2rotvec(pose)

            np.save(f'../../../dance_poses/pose_{pose_num:04d}.npy', pose)
            pose_num = pose_num + 1

#     result = model.detect(input_img)

#     cam = np.squeeze(result['cam'].numpy())[:3]
#     vertices = np.squeeze(result['vertices'].numpy())
#     pose = np.squeeze(result['pose'].numpy())



# # Assuming joints is your 24x3x3 array
    # num_joints = pose.shape[0]
    # axis_angle_params = np.zeros((num_joints, 3))

    # for i in range(num_joints):
    #     # Extract the rotation matrix for the i-th joint
    #     rotation_matrix = pose[i]

    #     # Convert the rotation matrix to axis-angle representation
    #     rotation = Rotation.from_matrix(rotation_matrix)
    #     axis_angle_params[i] = rotation.as_rotvec()

#     np.save('vertices.npy', vertices)
#     np.save('pose.npy', axis_angle_params)

#     joints = np.squeeze(result['kp2d'].numpy())
#     joints = ((joints + 1) * 0.5) * params['img_size']

#     renderer = TrimeshRenderer()
#     visualize(renderer, original_img, params, vertices, cam, joints)
