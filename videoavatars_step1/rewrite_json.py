import os
import json
from tqdm import tqdm

input_folder_path = '../keypoints_json/'
output_folder_path = '../jsons/'

# 获取文件夹中所有文件的列表

# 遍历文件列表
for file_name in tqdm(sorted(os.listdir(input_folder_path))):
    # 确保文件是 JSON 文件
    if file_name.endswith('.json'):
        file_path = os.path.join(input_folder_path, file_name)
        
        # 打开并读取 JSON 文件
        with open(file_path, 'r') as json_file:
            # 解析 JSON 数据
            json_data = json.load(json_file)
            body25 = json_data['people'][0]['pose_keypoints_2d']
            coco = []
            delete_list = [8, 19, 20, 21, 22, 23, 24]
            for i in range(int(len(body25) / 3)):
                if i not in delete_list:
                    coco.append(body25[3 * i])
                    coco.append(body25[3 * i + 1])
                    coco.append(body25[3 * i + 2])
            json_data['people'][0]['pose_keypoints_2d'] = coco

            output_file_path = os.path.join(output_folder_path, file_name)
            with open(output_file_path, 'w') as output_file:
                json.dump(json_data, output_file)
                    
