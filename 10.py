# emergent_parallel_thinker.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np

# (PulseTGAU 类的定义，保持不变)
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
# =================== 模块2: EmergentParallelThinker  ===============
# ==============================================================================

# 【新】我们先，定义一个，“层”的“内部核心”子模块
class LayerCore(nn.Module):
    def __init__(self, network_size: int, is_first_layer: bool, input_size: int):
        super().__init__()
        self.network_size = network_size
        self.is_first_layer = is_first_layer
        
        # 将，所有的“零件”，都，作为，这个“子模块”的“合法”成员
        if self.is_first_layer:
            self.input_interface = nn.Linear(input_size, self.network_size)
        else:
            self.inter_layer_weights = nn.Linear(self.network_size, self.network_size)
            
        self.recurrent_weights = nn.Parameter(
            torch.randn(self.network_size, self.network_size) * (1.0 / np.sqrt(self.network_size))
        )
        self.neuron_core = PulseTGAU(feature_dim=self.network_size)
        self.layer_norm = nn.LayerNorm(self.network_size)
        
class EmergentParallelThinker(nn.Module):
    def __init__(self, input_size=28, num_rps=5, rp_size=72, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.network_size = num_rps * rp_size
        self.num_layers = num_layers

       
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            is_first = (i == 0)
            self.layers.append(LayerCore(self.network_size, is_first, self.input_size))
        
        self.classifier = nn.Linear(self.network_size, 10)

    def forward(self, x: torch.Tensor, total_computation_time: int):
        batch_size, seq_len, _ = x.shape
        
        potentials = [None] * self.num_layers
        spikes_list = [torch.zeros(batch_size, self.network_size, device=x.device) for _ in range(self.num_layers)]
        
        for t in range(total_computation_time):
            next_spikes_list = [None] * self.num_layers
            
            for i in range(self.num_layers):
                layer = self.layers[i]
                
                non_recurrent_input = None
                if layer.is_first_layer:
                    external_input = layer.input_interface(x[:, t % seq_len, :])
                    non_recurrent_input = external_input
                else:
                    lower_layer_spikes = spikes_list[i-1]
                    non_recurrent_input = layer.inter_layer_weights(lower_layer_spikes)
                
                recurrent_signal = F.linear(spikes_list[i], layer.recurrent_weights)
                
                total_current = non_recurrent_input + recurrent_signal
                total_current = layer.layer_norm(total_current)
                
                new_spikes, new_potential = layer.neuron_core(total_current, potentials[i])
                
                next_spikes_list[i] = new_spikes
                potentials[i] = new_potential
                
            spikes_list = next_spikes_list

        final_spikes = spikes_list[-1]
        logits = self.classifier(final_spikes)
        return logits
    


# ==============================================================================
# ================= 模块3: 训练与评估主程序  =================
# ==============================================================================

def main():
    # --- 超参数配置 ---
    EPOCHS = 100
    BATCH_SIZE = 128
    MAX_LEARNING_RATE = 0.001
    RP_SIZE = 72
    NUM_RPS = 4
    NUM_LAYERS = 2 # 我们来测试一个“双层”的并行思考者
    
    # 【新】总计算时间 (Total Computation Time)
    # 我们先用一个，和之前，成功的串行模型，差不多的总时长
    TOTAL_COMPUTATION_TIME = 29 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动【并行思考者 (EmergentParallelThinker)】训练程序 ---")
    print(f"--- 任务：逐行扫描MNIST序列 ---")
    print(f"--- 总计算时间: {TOTAL_COMPUTATION_TIME} 步 | 层数: {NUM_LAYERS} ---")
    print(f"设备: {device}\n")

    # --- 数据加载 ---
    print("正在加载 MNIST 数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    # 增加 num_workers 和 pin_memory 可以略微提升数据加载速度
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print("数据集加载完毕。\n")

    # --- 模型、损失函数与优化器初始化 ---
    print("正在初始化 EmergentParallelThinker 模型...")
    model = EmergentParallelThinker(
        num_rps=NUM_RPS,
        rp_size=RP_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MAX_LEARNING_RATE)
    
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
        
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.squeeze(1) # [B, 1, 28, 28] -> [B, 28, 28]
            
            optimizer.zero_grad()
            output = model(data, total_computation_time=TOTAL_COMPUTATION_TIME)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 评估阶段 ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.squeeze(1)
                output = model(data, total_computation_time=TOTAL_COMPUTATION_TIME)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']


        # --- 打印周期报告 ---
        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        print(f"耗时: {epoch_time:.2f}s | 当前学习率: {current_lr:.6f}")
        print(f"训练损失: {avg_train_loss:.4f} | 测试准确率: {accuracy:.2f}%")
        print("-" * (21 + len(str(EPOCHS)) + len(str(epoch))))

    print("\n训练完成！")
    print(f"最终模型在测试集上的准确率为: {accuracy:.2f}%")


# --- 主程序入口 ---
if __name__ == "__main__":
    main()