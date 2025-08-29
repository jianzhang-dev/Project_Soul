# emergent_fusion.py
# 一个完整的、实现了“思考深度”课程学习的训练脚本。


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import math

# =========================================================
# =================== 模块1: PulseTGAU ====================
# =========================================================
class PulseTGAU(nn.Module):
 
    def __init__(self, feature_dim: int):
        super().__init__()
        self.threshold = nn.Parameter(torch.randn(feature_dim))
        self.intrinsic_value = nn.Parameter(torch.randn(feature_dim))
        self.feature_dim = feature_dim
        self.activation = nn.SiLU()
        self.gate_steepness = nn.Parameter(torch.ones(feature_dim) * 5.0)
        self.reset_scale = nn.Parameter(torch.ones(feature_dim) * 0.5)
        self.leakage_factor = 0.95

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
        potential = potential - self.reset_scale * reset_amount
        return output, potential

# ==============================================================================
# =================== 模块2: EmergentResonator  =============
# ==============================================================================
class EmergentResonator(nn.Module):
    def __init__(self, input_size=28, num_rps=5, rp_size=72):
        super().__init__()
        self.input_size = input_size
        self.num_rps = num_rps
        self.rp_size = rp_size
        self.network_size = num_rps * rp_size
        
        self.input_interface = nn.Linear(self.input_size, self.network_size)
        
        self.input_gate_weights = nn.Linear(self.input_size, self.network_size)
        
        self.recurrent_weights = nn.Parameter(
            torch.randn(self.network_size, self.network_size) * (1.0 / np.sqrt(self.network_size))
        )
        self.neuron_core = PulseTGAU(feature_dim=self.network_size)
        self.layer_norm = nn.LayerNorm(self.network_size)
        self.classifier = nn.Linear(self.network_size, 10)

    def forward(self, x: torch.Tensor, reasoning_steps: int):
        batch_size, seq_len, _ = x.shape
        potential = None
        current_spikes = torch.zeros(batch_size, self.network_size, device=x.device)
        
        # 1. “风”吹拂的阶段 (28个时间步)
        for t in range(seq_len):
            step_input = x[:, t, :]
            
            input_gate = torch.sigmoid(self.input_gate_weights(step_input))
            
            
            weighted_input_signal = self.input_interface(step_input)
            gated_input_signal = weighted_input_signal * input_gate 
            
            
            weighted_recurrent_signal = F.linear(current_spikes, self.recurrent_weights)
            
            total_current = gated_input_signal + weighted_recurrent_signal
            
            total_current = self.layer_norm(total_current)
            current_spikes, potential = self.neuron_core(total_current, potential)

        # 2. “风停了”，开始纯粹的“回响”
        for _ in range(reasoning_steps):
            weighted_recurrent_signal = F.linear(current_spikes, self.recurrent_weights)
            total_current = weighted_recurrent_signal
            total_current = self.layer_norm(total_current)
            current_spikes, potential = self.neuron_core(total_current, potential)
            
        logits = self.classifier(current_spikes)
        return logits

# ==============================================================================
# =================== 模块3: 训练与评估主程序  ==================
# ==============================================================================
def main():
    # --- 超参数配置  ---
    TOTAL_EPOCHS = 800
    BATCH_SIZE = 128
    MAX_LEARNING_RATE = 0.001
    RP_SIZE = 72
    NUM_RPS = 5
    
    # 【思考深度的“课程学习”参数 】(测试用，不必要)
    INITIAL_REASONING_STEPS = 1
    RAMP_UP_START_EPOCH = 20
    RAMP_UP_END_EPOCH = 800
    MAX_REASONING_STEPS = 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动【涌现式共鸣器 · 神性融合版】 ---")
    print(f"--- 总计 {TOTAL_EPOCHS} Epochs, 思考深度将从 {INITIAL_REASONING_STEPS} 步动态增长至 {MAX_REASONING_STEPS} 步 ---")
    print(f"设备: {device}\n")

    # --- 数据加载 (完全保持不变) ---
    print("正在加载 MNIST 数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("数据集加载完毕。\n")

    # --- 模型、损失函数与优化器初始化 (完全保持不变) ---
    print("正在初始化 EmergentResonator 模型...")
    model = EmergentResonator(num_rps=NUM_RPS, rp_size=RP_SIZE).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MAX_LEARNING_RATE)
    
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=MAX_LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=TOTAL_EPOCHS,
        pct_start=0.25
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型初始化完毕。总可训练参数: {num_params:,}\n")

    # --- 训练与评估循环 (完全保持不变) ---
    for epoch in range(1, TOTAL_EPOCHS + 1):
        
        current_reasoning_steps = 0
        if epoch < RAMP_UP_START_EPOCH:
            current_reasoning_steps = INITIAL_REASONING_STEPS
        elif epoch >= RAMP_UP_END_EPOCH:
            current_reasoning_steps = MAX_REASONING_STEPS
        else:
            progress = (epoch - RAMP_UP_START_EPOCH) / (RAMP_UP_END_EPOCH - RAMP_UP_START_EPOCH)
            current_reasoning_steps = int(INITIAL_REASONING_STEPS + progress * (MAX_REASONING_STEPS - INITIAL_REASONING_STEPS))

        start_time = time.time()
        model.train()
        total_train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.squeeze(1)
            
            optimizer.zero_grad()
            output = model(data, reasoning_steps=current_reasoning_steps)
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
                output = model(data, reasoning_steps=current_reasoning_steps)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"--- Epoch {epoch}/{TOTAL_EPOCHS} | 思考步数: {current_reasoning_steps} ---")
        print(f"耗时: {epoch_time:.2f}s | 当前学习率: {current_lr:.6f}")
        print(f"训练损失: {avg_train_loss:.4f} | 测试准确率: {accuracy:.2f}%")
        print("-" * (31 + len(str(TOTAL_EPOCHS)) + len(str(epoch)) + len(str(current_reasoning_steps))))

    print("\n【进化之旅 · 终章】")
    print(f"最终模型在测试集上的准确率为: {accuracy:.2f}%")

# --- 主程序入口 ---
if __name__ == "__main__":
    main()
