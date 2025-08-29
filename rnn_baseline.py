# rnn_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np

# 标准RNN基线模型
class RNNBaseline(nn.Module):
    def __init__(self, hidden_size=192, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 视觉前端 (与EmergentVision保持一致)
        self.vision_frontend = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28 -> 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_size)
        )
        
        # RNN核心 (LSTM)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        # 编码图像到特征向量
        features = self.vision_frontend(x)  # [B, 360]
        
        # 将静态特征转换为时序输入 (重复12次)
        seq_features = features.unsqueeze(1).repeat(1, 12, 1)  # [B, 12, 360]
        
        # RNN处理
        rnn_out, (h_n, c_n) = self.rnn(seq_features)
        
        # 使用最后时刻的隐藏状态分类
        logits = self.classifier(h_n[-1])  # [B, 10]
        return logits

def probe_rnn_memory(model, device, total_steps=1000):

    print("\n" + "="*65)
    print("⏳ 开始进行 RNN 长期记忆测试（仅输入一次，观察1000步）...")
    print("="*65)

    model.eval()
    batch_size = 1
    with torch.no_grad():
        # 1. 构造测试图像
        test_input = torch.randn(batch_size, 1, 28, 28, device=device)

        # 2. 提取特征（只做一次）
        initial_features = model.vision_frontend(test_input)  # [1, hidden_size]

        # 3. 初始化隐藏状态
        h = torch.zeros(model.num_layers, batch_size, model.hidden_size, device=device)
        c = torch.zeros(model.num_layers, batch_size, model.hidden_size, device=device)

        # 4. 存储活跃值
        activity_log = []

        # 5. 模拟 1000 步
        for t in range(total_steps):
            if t == 0:
                input_t = initial_features.unsqueeze(1)  # 第一步输入图像
            else:
                input_t = torch.zeros(batch_size, 1, model.hidden_size, device=device)

            # 推送输入
            _, (h, c) = model.rnn(input_t, (h, c))

            # 记录最后一层隐藏状态的 L1 活跃值
            activity = h[-1].abs().sum().item()
            activity_log.append(activity)

    # 6. 分析结果
    peak_activity = activity_log[0]
    mid_activity_mean = np.mean(activity_log[500:750])   # 中期（500~750）
    late_activity_mean = np.mean(activity_log[750:])     # 后期（750~999）
    final_activity = activity_log[-1]
    decay_ratio = final_activity / peak_activity if peak_activity > 0 else 0

    print(f"📈 活跃值日志 (前5步): {[f'{v:.2f}' for v in activity_log[:5]]}")
    print(f"📈 活跃值日志 (最后5步): {[f'{v:.2f}' for v in activity_log[-5:]]}")
    print(f"\n📊 超长期记忆分析:")
    print(f"  - 峰值活跃值 (t=0): {peak_activity:.2f}")
    print(f"  - 最终活跃值 (t=999): {final_activity:.2f}")
    print(f"  - 中期平均 (t=500~750): {mid_activity_mean:.2f}")
    print(f"  - 后期平均 (t=750~999): {late_activity_mean:.2f}")
    print(f"  - 衰减率 (final/initial): {decay_ratio*100:.1f}%")

    # 7. 判断长期稳定性
    if late_activity_mean > 0.1 and decay_ratio > 0.5:
        print("\n  [结论] 🟢 具备**超强长期记忆能力**！")
        if abs(mid_activity_mean - late_activity_mean) < 0.5:
            print("     ✅ 活跃值高度稳定 → 可能是数值锁定")
        else:
            print("     ⚠️  活跃值缓慢漂移 → 动态不稳定")
    elif late_activity_mean > 0.01:
        print("\n  [结论] 🟡 记忆持续衰减，未完全消失")
        print("     ⏳ 可能存在数值漂移或梯度饱和")
    else:
        print("\n  [结论] 🔴 长期记忆完全消失")
        print("     ❌ RNN 无法维持超长时间记忆")

    print("="*65)
    return activity_log
def main():
    # 超参数 (与EmergentVision保持一致)
    EPOCHS = 10
    BATCH_SIZE = 128
    MAX_LEARNING_RATE = 0.001
    HIDDEN_SIZE = 192
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- RNN基线模型训练 ---")
    print(f"设备: {device}\n")

    # 数据加载 (与EmergentVision保持一致)
    print("正在加载 Fashion-MNIST 数据集...")
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.FashionMNIST('data', train=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("数据集加载完毕。\n")

    # 模型初始化
    print("正在初始化 RNN 基线模型...")
    model = RNNBaseline(hidden_size=HIDDEN_SIZE).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MAX_LEARNING_RATE)
    
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=MAX_LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.2
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型初始化完毕。总可训练参数: {num_params:,}\n")

    # 训练与评估循环
    best_accuracy = 0.0
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # 评估阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'rnn_baseline_best.pth')
        
        # 打印周期报告
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Time: {epoch_time:.2f}s | "
              f"LR: {current_lr:.6f} | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Acc: {accuracy:.2f}%")

    print(f"\n训练完成！")
    print(f"最佳准确率: {best_accuracy:.2f}%")
    
    # 活跃值测试
    probe_rnn_memory(model, device)

if __name__ == "__main__":
    main()
