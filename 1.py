# emergent_resonator.py
# 一个完整的训练脚本，用于我们最终的、最纯粹的串行模型。
# 它实现了“风吹钟响”的“激发-共鸣”范式。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np


class PulseTGAU(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.threshold = nn.Parameter(torch.randn(feature_dim))
        self.intrinsic_value = nn.Parameter(torch.randn(feature_dim))
        self.feature_dim = feature_dim
        self.activation = nn.SiLU()
        self.leakage_factor = 0.95
        self.gate_steepness = nn.Parameter(torch.ones(feature_dim) * 5.0)

    def forward(self, x: torch.Tensor, potential: torch.Tensor = None):
        if potential is None:
            batch_size = x.size(0)
            device = x.device
            potential = torch.zeros(batch_size, self.feature_dim, device=device)
        
        potential = potential * self.leakage_factor
        potential = potential + x + self.intrinsic_value
        gated_potential = potential - self.threshold
        soft_gate = torch.sigmoid(gated_potential * self.gate_steepness.abs())
        output = self.activation(gated_potential) * soft_gate
        reset_amount = gated_potential * soft_gate
        potential = potential - reset_amount.detach()
        return output, potential

# ==============================================================================
# =================== 模块2: EmergentResonator (最终的串行模型) ==================
# ==============================================================================

class EmergentResonator(nn.Module):
    def __init__(self, input_size=28, num_rps=5, rp_size=72, 
                 connection_threshold=0.0, 
                 reasoning_steps=12):
        super().__init__()
        self.input_size = input_size
        self.num_rps = num_rps
        self.rp_size = rp_size
        self.network_size = num_rps * rp_size
        self.connection_threshold = connection_threshold
        self.reasoning_steps = reasoning_steps
        
        self.input_interface = nn.Linear(self.input_size, self.network_size)

        self.recurrent_weights = nn.Parameter(
            torch.randn(self.network_size, self.network_size) * (1.0 / np.sqrt(self.network_size))
        )
        self.neuron_core = PulseTGAU(feature_dim=self.network_size)
        self.layer_norm = nn.LayerNorm(self.network_size)
        self.classifier = nn.Linear(self.network_size, 10)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        
        potential = None
        current_spikes = torch.zeros(batch_size, self.network_size, device=x.device)
        
        # 1. “风”吹拂的阶段 (28个时间步)
        for t in range(seq_len):
            step_input = x[:, t, :]
            weighted_input_signal = self.input_interface(step_input)
            
            mask = (self.recurrent_weights.abs() > self.connection_threshold).float()
            effective_recurrent_weights = self.recurrent_weights * mask
            weighted_recurrent_signal = F.linear(current_spikes, effective_recurrent_weights)
            
            total_current = weighted_input_signal + weighted_recurrent_signal
            total_current = self.layer_norm(total_current)
            
            current_spikes, potential = self.neuron_core(total_current, potential)

        # 2. “风停了”，开始纯粹的“回响” (reasoning_steps 步)
        for _ in range(self.reasoning_steps):
            weighted_recurrent_signal = F.linear(current_spikes, self.recurrent_weights)
            
            total_current = weighted_recurrent_signal
            total_current = self.layer_norm(total_current)
            
            current_spikes, potential = self.neuron_core(total_current, potential)
            
        # 3. “听音辨物”
        logits = self.classifier(current_spikes)
        return logits


# ==============================================================================
# =================== 模块3: 训练与评估主程序 =====================================
# ==============================================================================

def main():
    # --- 超参数配置 ---
    EPOCHS = 100 #调成3和5可进行活跃度变化对比
    BATCH_SIZE = 128
    MAX_LEARNING_RATE = 0.001
    RP_SIZE = 72       # 每个RP的大小 (神经元数量)
    NUM_RPS = 5        # RP的数量
    REASONING_STEPS = 1 # “风停后”的纯粹思考步数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动【涌现式共鸣器 (Emergent Resonator)】训练程序 ---")
    print(f"--- 任务：逐行扫描MNIST序列 ---")
    print(f"设备: {device}\n")

    # --- 数据加载 (使用我们用于序列任务的配置) ---
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
    print("正在初始化 EmergentResonator 模型...")
    model = EmergentResonator(
        num_rps=NUM_RPS,
        rp_size=RP_SIZE,
        reasoning_steps=REASONING_STEPS
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MAX_LEARNING_RATE)
    
    # 使用 OneCycleLR 学习率调度器
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=MAX_LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.25
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
            # 我们需要将其变成 [B, 28, 28] (批次, 序列长度, 特征)
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