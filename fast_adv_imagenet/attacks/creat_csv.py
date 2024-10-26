import os
import pandas as pd

# 设置图片文件夹路径
image_folder = 'imagenet'

# 获取图片文件夹中的所有文件
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 创建一个 DataFrame
data = {
    'image_path': [os.path.join(image_folder, f) for f in image_files],
    'label': [0] * len(image_files)  # 假设标签都设置为0，可以根据需要修改
}

df = pd.DataFrame(data)

# 将 DataFrame 保存为 CSV 文件
csv_file = 'dev.csv'
df.to_csv(csv_file, index=False)

print(f"CSV 文件 '{csv_file}' 已创建。")
