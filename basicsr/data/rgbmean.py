import numpy as np
import cv2
import os

def calculate_dataset_rgb_mean(dataset_path):
    # 获取数据集下所有图像文件的路径
    image_files = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]

    # 初始化累计的RGB通道值
    total_r = 0
    total_g = 0
    total_b = 0

    # 遍历图像文件并累计RGB通道值
    for image_file in image_files:
        # 加载图像
        image = cv2.imread(image_file)
        # 将图像转换为RGB顺序
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 计算每个通道的总和
        total_r += np.sum(image[:, :, 0])
        total_g += np.sum(image[:, :, 1])
        total_b += np.sum(image[:, :, 2])

    # 计算RGB通道的均值
    num_pixels = len(image_files) * image.shape[0] * image.shape[1]
    rgb_mean = (total_r / num_pixels, total_g / num_pixels, total_b / num_pixels)

    return rgb_mean

# 数据集路径
dataset_path = "D:\celeba\data\CELEBA\TRAIN_HR"
# 计算RGB均值
mean_rgb = calculate_dataset_rgb_mean(dataset_path)
print("RGB Mean:", mean_rgb)