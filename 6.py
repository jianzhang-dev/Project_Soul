

# --- 导入必要的库 ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time


# --- 模块1: PulseTGAU ---
#  输入x -> 更新电位 -> 对比阈值 -> 脉冲输出)
class PulseTGAU(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.threshold = nn.Parameter(torch.randn(feature_dim))
        self.intrinsic_value = nn.Parameter(torch.randn(feature_dim))
        self.feature_dim = feature_dim
        self.activation = nn.SiLU()
        self.leakage_factor = 0.95
    
    def forward(self, x: torch.Tensor, potential: torch.Tensor = None):
        # 如果没有提供初始状态，创建新的零状态
        if potential is None:
            batch_size = x.size(0)
            device = x.device
            potential = torch.zeros(batch_size, self.feature_dim, device=device)
        
        # 电位更新
        potential = potential * self.leakage_factor
        potential = potential + x + self.intrinsic_value
        
        # 脉冲生成
        gated_potential = potential - self.threshold
        spike_mask = (gated_potential > 0).float()
        output = self.activation(gated_potential) * spike_mask
        
        # 电位重置
        reset_amount = gated_potential * spike_mask
        potential = potential - reset_amount.detach()
        
        return output, potential



# ==============================================================================
# =================== 最终、完整的 EmergentRNN 类定义 ==========================
# ==============================================================================

# (PulseTGAU 类的定义保持不变)

class EmergentVision(nn.Module):
    def __init__(self, num_rps=5, rp_size=64, img_size=28, 
                 connection_threshold=0.01, 
                 iteration_steps=12): # 【新】内部思考的步数
        super().__init__()
        self.num_rps = num_rps
        self.rp_size = rp_size
        self.network_size = num_rps * rp_size
        self.connection_threshold = connection_threshold
        self.iteration_steps = iteration_steps # 存储迭代步数

        # 1. 【新】视觉编码器 (The Eye)
        #    一个简单的全连接层，将扁平化的图像 (28*28=784) 映射到网络的初始状态。
        self.vision_frontend = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # <-- 新增批量归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28 -> 14x14

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # <-- 新增批量归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14 -> 7x7

            # Flatten and project
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, self.network_size) # 输入维度也需要更新
        )
        # 2. 循环权重 (Recurrent Weights) 
        self.recurrent_weights = nn.Parameter(
             torch.randn(self.network_size, self.network_size) * (1.0 / np.sqrt(self.network_size))
        )

        # 3. 脉冲神经元层 (The Neuron Core) - 保持不变！
        self.neuron_core = PulseTGAU(feature_dim=self.network_size)

        # 4. 层归一化 (LayerNorm) - 保持不变！
        self.layer_norm = nn.LayerNorm(self.network_size)
        
        # 5. 分类器 (Classifier) - 保持不变！
        self.classifier = nn.Linear(self.network_size, 10)

    def forward(self, x: torch.Tensor):
        """
        处理一个MNIST图像批次。
        x 的形状: [batch_size, 1, 28, 28]
        """
        batch_size = x.shape[0]
        
        # a. 【新】一次性编码整个图像
        #    首先将图像从 [B, 1, 28, 28] 扁平化为 [B, 784]
        initial_current = self.vision_frontend(x)

        # b. 【新】内部迭代循环 (The "Thinking" Process)
        #    初始化状态
        potential = None
        current_spikes = torch.zeros(batch_size, self.network_size, device=x.device)
        
        #    开始固定步数的“思考”
        for step in range(self.iteration_steps):
            # i. 计算来自网络上一步状态的“回响”
            mask = (self.recurrent_weights.abs() > self.connection_threshold).float()
            effective_recurrent_weights = self.recurrent_weights * mask
            weighted_recurrent_signal = F.linear(current_spikes, effective_recurrent_weights)

            # ii. 总电流 = 初始想法(只在第一步有影响) + 内部回响
            #     我们只在第一步注入外部信息，之后让网络自己“思考”。
            #     这是一种常见的做法，也可以设计成每一步都注入。
            if step == 0:
                total_current = initial_current + weighted_recurrent_signal
            else:
                total_current = weighted_recurrent_signal
            
            # iii. 稳定并激活
            total_current = self.layer_norm(total_current)
            current_spikes, potential = self.neuron_core(total_current, potential)

        # c. 最终决策
        #    在经过多轮“思考”后，用最终的网络状态进行分类。
        final_representation = current_spikes
        logits = self.classifier(final_representation)
        
        return logits
    
    # (在 EmergentRNN 类的定义内部，替换掉旧的 print_connection_stats 函数)

    def print_connection_stats(self):
        """
        【增强版】一个辅助函数，用于打印当前网络连接的统计信息，
        并监控自发趋近于0的连接数量。
        """
        with torch.no_grad():
            # --- 旧功能：基于手动阈值的统计 (我们仍然保留它用于对比) ---
            weights_abs = self.recurrent_weights.abs()
            mask = (weights_abs > self.connection_threshold).float()
            
            total_connections = weights_abs.numel()
            active_connections = mask.sum().item()
            sparsity_manual = 100 * (1 - active_connections / total_connections)
            
            rp_size = self.rp_size
            num_rps = self.num_rps
            intra_rp_connections = 0
            inter_rp_connections = 0
            
            for i in range(num_rps):
                for j in range(num_rps):
                    sub_matrix = mask[i*rp_size:(i+1)*rp_size, j*rp_size:(j+1)*rp_size]
                    if i == j:
                        intra_rp_connections += sub_matrix.sum().item()
                    else:
                        inter_rp_connections += sub_matrix.sum().item()

            print(f"    - [手动剪枝统计 @ 阈值={self.connection_threshold}]")
            print(f"    - 活跃连接: {int(active_connections)} / {total_connections} (稀疏度: {sparsity_manual:.2f}%)")
            print(f"    - RP内部: {int(intra_rp_connections)} | RP之间: {int(inter_rp_connections)}")

            # --- 【新功能】: 监控自发演化中趋近于0的连接 ---
            print(f"    - [自发稀疏性统计 (权重绝对值)]")
            
            thresholds_to_check = [0.01, 0.001, 0.0001]
            for thr in thresholds_to_check:
                # 计算权重绝对值小于当前检查阈值的连接数量
                near_zero_count = (weights_abs < thr).sum().item()
                percentage = 100 * near_zero_count / total_connections
                print(f"    - 权重 < {thr}: {near_zero_count} 个 ({percentage:.2f}%)")



def probe_memory_vortex(model, device, probe_duration=10000):
   
    print("\n" + "="*80)
    print("🌀⚡🌌 开始进行“脉冲注入”实验：探测10000步内的活跃值波动...")
    print("="*80)

    model.eval()
    batch_size = 1
    
    with torch.no_grad():
        # 1. 初始化
        potential = None
        current_spikes = torch.zeros(batch_size, model.network_size, device=device)
        activity_log = []

        # 2. 注入一次脉冲
        impulse_image = torch.randn(batch_size, 1, 28, 28, device=device)
        initial_current = model.vision_frontend(impulse_image)

        # 3. 模拟 10,000 步
        print(f"⏳ 开始模拟 10,000 步... (请耐心等待)")
        for t in range(probe_duration):
            if t % 1000 == 0 and t > 0:
                print(f"  🚶 已完成 {t} 步...")

            # a. 循环信号
            mask = (model.recurrent_weights.abs() > model.connection_threshold).float()
            effective_recurrent_weights = model.recurrent_weights * mask
            weighted_recurrent_signal = F.linear(current_spikes, effective_recurrent_weights)

            # b. 总电流（仅 t=0 注入）
            total_current = initial_current + weighted_recurrent_signal if t == 0 else weighted_recurrent_signal
            
            # c. 稳定并激活
            if hasattr(model, 'layer_norm'):
                total_current = model.layer_norm(total_current)
            
            current_spikes, potential = model.neuron_core(total_current, potential)

            # d. 记录活跃值
            total_activity = torch.sum(current_spikes.abs()).item()
            activity_log.append(total_activity)

    # --- 分析结果 ---
    print(f"✅ 实验完成！共模拟 {probe_duration} 个时间步。")
    print(f"📈 活跃值日志 (前5步): {[f'{v:.2f}' for v in activity_log[:5]]}")
    print(f"📈 活跃值日志 (最后5步): {[f'{v:.2f}' for v in activity_log[-5:]]}")

    peak_activity = max(activity_log[:2])
    late_activity_mean = np.mean(activity_log[5000:])  # 后5000步平均
    late_activity_std = np.std(activity_log[5000:])
    final_activity = activity_log[-1]
    decay_ratio = final_activity / peak_activity if peak_activity > 0 else 0

    min_late = min(activity_log[5000:])
    max_late = max(activity_log[5000:])

    print(f"\n📊 终极记忆分析 (t=5000~9999):")
    print(f"  - 峰值活跃值 (t=0~1): {peak_activity:.2f}")
    print(f"  - 最终活跃值 (t=9999): {final_activity:.2f}")
    print(f"  - 后5000步平均: {late_activity_mean:.2f}")
    print(f"  - 后5000步波动 (std): {late_activity_std:.2f}")
    print(f"  - 范围: [{min_late:.2f}, {max_late:.2f}]")
    print(f"  - 衰减率 (final/peak): {decay_ratio*100:.1f}%")

    # 判断
    if late_activity_mean > 0.1 and decay_ratio > 0.5:
        if late_activity_std > 1.0:
            print("     ✅✅✅ 活跃值持续震荡")
        else:
            print("\n  [结论] 🟢 存在长期记忆，但趋于静态")
            print("     ⚠️  可能已进入数值锁定态，失去动态性")
    elif late_activity_mean > 0.01:
        print("\n  [结论] 🟡 记忆持续衰减，未完全消失")
        print("     ⏳ 可能在 20,000 步内消失")
    else:
        print("\n  [结论] 💀 超长期记忆完全消失")
        print("     ❌ 网络未能维持信息循环")

    print("="*80)
    return activity_log
# ...

# ==============================================================================
# =================== 【完整更新版】的 main 函数 ===============================
# ==============================================================================

def main():
    # --- 1. 超参数与设备配置 ---
    EPOCHS = 100 
    BATCH_SIZE = 128
    RP_SIZE = 72
    MAX_LEARNING_RATE = 0.001 
    ITERATION_STEPS = 12 
    
    CONNECTION_THRESHOLD = 0.0 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 启动【视觉网络】-> 挑战 Fashion-MNIST (使用OneCycleLR) ---")
    print(f"设备: {device}\n")

    # --- 2. 数据加载 ---
    print("正在加载 Fashion-MNIST 数据集 (带数据增强)...")
    
    # 【新】为训练集创建一个带数据增强的 transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # 50%的概率水平翻转
        transforms.RandomRotation(10), # 在-10到+10度之间随机旋转
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 【不变】测试集永远不使用数据增强，以保证评估的公正性
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 应用新的 transform
    train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.FashionMNIST('data', train=False, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("数据集加载完毕。\n")

       # --- 3. 模型、损失函数与优化器初始化 ---
    print("正在初始化 EmergentVision 模型...")
    
    # a. 先在CPU上创建基础模型
    model = EmergentVision(
        rp_size=RP_SIZE,
        connection_threshold=CONNECTION_THRESHOLD,
        iteration_steps=ITERATION_STEPS
    )
    

    model.vision_frontend = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, model.network_size) # 使用 model.network_size 确保一致性
    )

    # c. 【核心修正】最后，将整个、完整的、替换好组件的模型，一次性移动到GPU
    model.to(device)
    
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

    # --- 4. 训练与评估循环 ---
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
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
            
            # 【核心修改】OneCycleLR需要在每个batch结束后都更新
            scheduler.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
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

          # --- 打印周期报告 (我们可以增加打印当前学习率) ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        print(f"耗时: {epoch_time:.2f}s | 当前学习率: {current_lr:.6f}")
        print(f"训练损失: {avg_train_loss:.4f} | 测试准确率: {accuracy:.2f}%")
        if hasattr(model, 'print_connection_stats'):
            model.print_connection_stats()
        print("-" * (21 + len(str(EPOCHS)) + len(str(epoch))))

    print("\n训练完成！")
    print(f"最终模型在 Fashion-MNIST 测试集上的准确率为: {accuracy:.2f}%")
    



    # 【新增这行】在所有事情都做完后，调用诊断函数
    probe_memory_vortex(model, device)

# --- 主程序入口 ---
if __name__ == "__main__":
    main()