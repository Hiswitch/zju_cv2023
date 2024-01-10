import numpy as np
import cPickle as pkl
from src.smpl import Smpl
from src import mesh
from tqdm import tqdm
import os
import cv2

with open('./models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl', 'rb') as fp:
        m = pkl.load(fp)
with open('./data/models/consensus.pkl', 'rb') as fp:
        consensus_data = pkl.load(fp)

m = Smpl(m)

input_folder_path = './data/poses/'
vt = np.load('assets/basicModel_vt.npy')
ft = np.load('assets/basicModel_ft.npy')
model_num = 0
pose_0 = 0
for filename in tqdm(sorted(os.listdir(input_folder_path))):
        if filename.endswith(".npy"):
                pose = np.load(os.path.join(input_folder_path, filename))
                pose = pose.reshape(72)

                angle = np.pi
                add_rmtx = cv2.Rodrigues(np.array([0, 0, -angle], dtype='float32'))[0]
                root_rmtx = cv2.Rodrigues(pose[:3])[0]
                new_root_rmtx = np.dot(add_rmtx, root_rmtx)
                pose[:3] = cv2.Rodrigues(new_root_rmtx)[0][:, 0]
                add_rmtx = cv2.Rodrigues(np.array([0, -angle, 0], dtype='float32'))[0]
                root_rmtx = cv2.Rodrigues(pose[:3])[0]
                new_root_rmtx = np.dot(add_rmtx, root_rmtx)
                pose[:3] = cv2.Rodrigues(new_root_rmtx)[0][:, 0]

                
                m.pose[:] = pose
                m.betas[:] = consensus_data['betas']
                m.v_personal[:] = consensus_data['v_personal']
                m.trans[:] = 0

                outmesh_path = './output/models/model_' + str(model_num).zfill(4) + '.obj'
                mesh.write(outmesh_path, m.r, m.f, vt=vt, ft=ft)

                model_num = model_num + 1
