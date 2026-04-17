import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm
import shutil
from PIL import ImageFile
from torch.amp import autocast, GradScaler

ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':

    # 1. 基础配置与路径设定
    BATCH_SIZE = 32  # 爆内存就调小
    NUM_EPOCHS = 30  # 其实20就差不多
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用的计算设备: {device}")

    stamp = time.strftime("%m%d-%H%M%S")
    out_dir = os.path.join('./out', stamp)
    os.makedirs(out_dir, exist_ok=True)

    train_dir = './dataset/train'
    test_dir = './dataset/val'

    # 2. 数据增强与加载
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, transform=transform_train)
    vali_dataset = ImageFolder(test_dir, transform=transform_val)

    # num_workers 根据自己需要调整
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                              pin_memory=True)
    vali_loader = DataLoader(dataset=vali_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"识别类别映射: {train_dataset.class_to_idx}")

    # 3. 动态计算类别权重
    class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in train_dataset.classes]
    total_samples = sum(class_counts)
    class_weights = [total_samples / c for c in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"自动计算的类别权重: {class_weights}")

    # 4. 模型、损失函数、优化器与加速器
    model = torchvision.models.resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler()

    # 5. 训练循环
    print('开始训练')
    best_acc = -1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_total = 0.0
        train_acc_total = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")
        for data, labels in train_bar:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(data)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = outputs.argmax(dim=1)
            acc = (pred == labels).float().mean()

            train_loss_total += loss.item()
            train_acc_total += acc.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        model.eval()
        vali_loss_total = 0.0
        vali_acc_total = 0.0

        with torch.no_grad():
            for data, labels in tqdm(vali_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]"):
                data, labels = data.to(device), labels.to(device)
                with autocast('cuda'):
                    outputs = model(data)
                    loss = criterion(outputs, labels)

                pred = outputs.argmax(dim=1)
                acc = (pred == labels).float().mean()

                vali_loss_total += loss.item()
                vali_acc_total += acc.item()

        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_acc = train_acc_total / len(train_loader)
        avg_val_acc = vali_acc_total / len(vali_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f'result | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f}')

        if avg_val_acc > best_acc:
            print(f'验证集准确率由 {best_acc:.4f} 提升至 {avg_val_acc:.4f}，正在保存模型')
            best_acc = avg_val_acc
            save_path = os.path.join(out_dir, f'epoch_{epoch + 1}_acc_{avg_val_acc:.4f}.pth')
            best_path = os.path.join(out_dir, 'model_best.pth')

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, save_path)
            shutil.copyfile(save_path, best_path)

    print(f"\n训练完成,最好的一版模型存放在 {os.path.join(out_dir, 'model_best.pth')}")