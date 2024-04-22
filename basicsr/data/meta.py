import os
from PIL import Image

folder_path = r'D:\dataset\celeba\data\CELEBA\rnhrp1'  # 替换为你的文件夹路径
output_file = 'meta_info/cele10p1_info.txt'
# 遍历文件夹中的所有文件
with open(output_file, 'w') as f:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 检查文件是否为图片
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 打开图像文件
                image = Image.open(file_path)

                # 提取图像的尺寸信息
                width, height = image.size
                channels = len(image.getbands())
                # 输出图像信息
                f.write(f"{filename} ({width}, {height}, {channels})\n")

                # 关闭图像文件
                image.close()

            except (IOError, SyntaxError) as e:
                print(f"处理文件 {filename} 时出错：{str(e)}")