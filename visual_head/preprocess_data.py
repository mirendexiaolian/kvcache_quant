import os
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw

# 初始化或加载数据集
dataset_path = "/path/to/datasets/synthdog-en-detection/data"
data = load_dataset(dataset_path, split="train")

# 设定图像和JSON文件存放位置
image_folder = "/path/to/datasets/synthdog-en/data"
json_folder = "/path/to/datasets/synthdog-en"

# 检查并创建 image_folder
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
    print(f"Directory '{image_folder}' created.")
else:
    print(f"Directory '{image_folder}' already exists.")

# 检查并创建 json_folder
if not os.path.exists(json_folder):
    os.makedirs(json_folder)
    print(f"Directory '{json_folder}' created.")
else:
    print(f"Directory '{json_folder}' already exists.")


converted_data = []
# 开始处理数据
for id, da in enumerate(tqdm(data)):
    json_data = {}
    json_data["id"] = id
    if da["image"] is not None:
        json_data["image_name"] = f"/path/to/datasets/synthdog-en/data/{id}.jpg"
        da["image"].save(os.path.join(image_folder, json_data["image_name"]))
    json_data["2_coord"] = da["2_coord"]
    converted_data.append(json_data)
    
with open("/path/to/datasets/synthdog-en/synthdog-en.json", "w") as f:
    json.dump(converted_data, f, indent=4)

print("数据处理完毕！")