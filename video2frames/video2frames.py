import cv2
import os
# 视频文件路径
video_path = '../dance.mp4'

# 输出帧图像的文件夹路径
output_folder = '../frames'

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频的帧速率
fps = cap.get(cv2.CAP_PROP_FPS)

# 读取视频帧
success, frame = cap.read()
count = 0

while success:
    # 构建输出帧的文件路径
    frame_path = os.path.join(output_folder, f"frame_{count:04d}.png")

    # 保存帧图像
    cv2.imwrite(frame_path, frame)

    # 读取下一帧
    success, frame = cap.read()
    count += 1

# 关闭视频捕获对象
cap.release()
