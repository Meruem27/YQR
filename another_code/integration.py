import os
import shutil

# 源文件夹路径列表
source_folders = [
    'E:/yqr_Project/2024.9.30/dataset/Triple'
]

# 目标文件夹路径
destination_folder = 'E:/yqr_Project/2024.9.30/dataset/All_animal'  # 请替换为你的目标文件夹路径

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 计数器，从1开始
counter = 1

# 遍历每个源文件夹
for source_folder in source_folders:
    # 获取源文件夹中的所有文件
    files = os.listdir(source_folder)

    # 遍历所有文件
    for file in files:
        # 检查文件扩展名是否为 .txt
        if file.endswith('.txt'):
            # 生成新的文件名，格式为 000001.txt
            new_name = f'{counter:04d}.txt'  # 使用六位数字格式
            # 构造源文件的完整路径
            source_file = os.path.join(source_folder, file)
            # 构造目标文件的完整路径
            destination_file = os.path.join(destination_folder, new_name)
            # 复制文件到目标文件夹并重命名
            shutil.copy2(source_file, destination_file)
            # 计数器加1
            counter += 1

print("文件复制完成！")
