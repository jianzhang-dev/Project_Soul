# classical_rnn_for_sequence.py
# 一个完整的、用于处理“逐行MNIST”序列任务的经典RNN(LSTM)基线模型，用作对比

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np

# ==============================================================================
# =================== 1. 经典RNN(LSTM)模型定义 =================================
# ==============================================================================

class SequentialLSTM(nn.Module):
    """
    一个经典的、多层的LSTM模型，专门用于处理逐行扫描的MNIST序列。
    """
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        # LSTM核心:
        # - input_size=28: 每个时间步(每一行)输入28个像素点。
        # - hidden_size=128: 内部记忆单元的维度。
        # - num_layers=2: 堆叠两层LSTM，以增加模型的深度和表达能力。
        # - batch_first=True: 让输入数据的维度顺序是 [批次, 序列长度, 特征]，更符合直觉。
        # - dropout: 在多层LSTM之间加入dropout，防止过拟合。
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 分类器:
        # 使用LSTM在处理完整个序列后，最后一个时间步的隐藏状态，来进行分类。
        self.classifier = nn.Linear(hidden_size, 10) # 10个类别 (0-9)
        
    def forward(self, x: torch.Tensor):
        # x的输入形状: [批次大小, 序列长度, 输入特征]
        # 对于我们的任务，即 [批次大小, 28, 28]

        # LSTM会返回三个东西：
        # 1. output: 包含了序列中每一个时间步的隐藏状态的集合。
        # 2. h_n: 只包含序列最后一个时间步的、每一层的隐藏状态。
        # 3. c_n: 只包含序列最后一个时间步的、每一层的细胞状态。
        output, (h_n, c_n) = self.lstm(x)
        
        # 我们通常使用最后一层、最后一个时间步的隐藏状态 h_n[-1] 作为整个序列的“总结”。
        # h_n 的形状是 [层数, 批次大小, 隐藏层维度]
        final_hidden_state = h_n[-1]
        
        # 将这个“总结”送入分类器，得到最终的 logits
        logits = self.classifier(final_hidden_state)
        
        return logits

# ==============================================================================
# =================== 2. 训练与评估主程序 =====================================
# ==============================================================================

def main():
    # --- 超参数配置 ---
    EPOCHS = 50
    BATCH_SIZE = 128
    MAX_LEARNING_RATE = 0.001
    HIDDEN_SIZE = 128  # LSTM内部记忆单元的大小
    NUM_LAYERS = 2     # LSTM的层数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动【经典LSTM模型】处理逐行MNIST序列任务 ---")
    print(f"设备: {device}\n")

    # --- 数据加载 ---
    print("正在加载 MNIST 数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("数据集加载完毕。\n")

    # --- 模型、损失函数与优化器初始化 ---
    print("正在初始化 SequentialLSTM 模型...")
    model = SequentialLSTM(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MAX_LEARNING_RATE)
    
    # 使用我们验证过有效的 OneCycleLR 学习率调度器
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=MAX_LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.25 # 将25%的时间用于预热
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型初始化完毕。总可训练参数: {num_params:,}\n")

    # --- 训练与评估循环 ---
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 关键的数据塑形：
            # DataLoader输出的data形状是 [B, 1, 28, 28]
            # 我们需要将其变成LSTM期望的 [B, 28, 28] (批次, 序列长度, 特征)
            data = data.squeeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.squeeze(1)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        print(f"耗时: {epoch_time:.2f}s | 当前学习率: {current_lr:.6f}")
        print(f"训练损失: {avg_train_loss:.4f} | 测试准确率: {accuracy:.2f}%")
        print("-" * (21 + len(str(EPOCHS)) + len(str(epoch))))

    print("\n训练完成！")
    print(f"最终模型在测试集上的准确率为: {accuracy:.2f}%")

# --- 主程序入口 ---
if __name__ == "__main__":
    main()