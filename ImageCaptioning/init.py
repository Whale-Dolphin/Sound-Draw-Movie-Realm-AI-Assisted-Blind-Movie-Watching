import torch
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "t5_base"  # 指定模型和配置文件所在的本地路径
model = T5ForConditionalGeneration.from_pretrained(model_path, verify=False)
tokenizer = T5Tokenizer.from_pretrained(model_path, verify=False)

image_folder = "input"  # 图像文件夹路径
output_csv = "../output/description.csv"  # 输出的.csv文件路径

# 检查图像文件夹路径是否存在
if not os.path.exists(image_folder):
    print("invalid path")
    exit()

# 加载并预处理图像的变换
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建一个空的描述列表
descriptions = []

# 遍历图像文件夹中的所有图像文件
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理.jpg和.png格式的图像文件
        # 构建图像路径
        image_path = os.path.join(image_folder, filename)

        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(model.device)

        # 生成图像描述
        with torch.no_grad():
            inputs = tokenizer.encode("generate caption: " + image_path, return_tensors="pt")
            inputs = inputs.to(model.device)
            outputs = model.generate(input_ids=inputs, decoder_input_ids=None, max_length=100)
            predicted_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 将图像描述添加到描述列表
        descriptions.append(predicted_caption)

# 创建一个包含图像路径和描述的DataFrame
data = pd.DataFrame({"path": os.listdir(image_folder), "description": descriptions})

# 将DataFrame写入.csv文件
data.to_csv(output_csv, index=False)
