import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# 1. 基础配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用的计算设备: {device}")

model_path = './out/0417-172404/model_best.pth'

folder_path = './dataset/val/others'

idx_to_class = {0: '奶龙', 1: '非奶龙'}

# 2. 重建模型与加载权重
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
model.eval()

# 3. 数据预处理
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. 批量读取图片并进行预测
if not os.path.exists(folder_path):
    print(f"找不到文件夹: '{folder_path}'，请先在代码同级目录下创建它！")
else:
    print(f"开始批量扫描 [{folder_path}] 文件夹")

    # 获取文件夹下所有文件
    all_files = os.listdir(folder_path)


    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

    if len(image_files) == 0:
        print("文件夹内无图片，或者没有找到支持的图片格式。")
    else:
        print(f"共找到 {len(image_files)} 张图片，开始分类：\n" + "-" * 50)

        # 遍历每一张图片
        for img_name in image_files:
            # 拼接出完整的图片路径
            img_path = os.path.join(folder_path, img_name)

            try:
                # 读取并处理当前图片
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)

                # 进行预测
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    max_prob, predicted_idx = torch.max(probabilities, 0)

                    predicted_idx = predicted_idx.item()
                    confidence = max_prob.item() * 100
                    predicted_name = idx_to_class.get(predicted_idx, '未知')

                # 打印这一张图片的结果
                print(f"文件: {img_name:<15} | 鉴定结果: {predicted_name:<6} | 置信度: {confidence:.2f}%")

            except Exception as e:
                print(f"处理图片 {img_name} 时出错: {e}")

        print("-" * 50)
        print("所有图片批量鉴定完毕！")