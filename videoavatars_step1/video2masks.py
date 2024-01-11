import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2

def video_split(video_path, out_path):
    vc = cv2.VideoCapture(video_path)  # 读入视频文件，命名cv
    n = 1  # 计数
    
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    
    timeF = 1  # 视频帧计数间隔频率
    
    i = 0
    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            i += 1
            print(i)
            cv2.imwrite(out_path + '/{}.jpg'.format(i), frame)  # 存储为图像
        n = n + 1
        cv2.waitKey(1)
    vc.release()

def segment_person(image_path, output_path):
    # 加载预训练的DeepLabV3模型
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()
 
    # 读取图片并转换
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
 
    #if torch.cuda.is_available():
    input_batch = input_batch.to('cpu')
    model.to('cpu')
 
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output = torch.argmax(output, dim=0).byte().cpu().numpy()
 
    # 人物语义分割标签 (在PASCAL VOC数据集中，人物用标签15表示)
    output_person = (output == 15)
 
    # 应用掩码
    mask = output_person.astype(np.uint8) * 255
    mask = Image.fromarray(mask)
    mask.save(output_path)
    # masked_image = Image.composite(input_image.size, Image.new('RGB', mask.size), mask)
 
    # masked_image.save(output_path)


# 使用方法
video_split("video.mp4", "./imgs")
# input_image_path = "2.jpg"
# output_image_path = "22.jpg"
# segment_person(input_image_path, output_image_path)