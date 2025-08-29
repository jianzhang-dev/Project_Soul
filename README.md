# Emergent Resonator: An RNN Architecture

---

### **A Note on the Project's Origin (关于项目起源)**

This project was developed by a non-professional enthusiast in collaboration with an AI assistant. It is presented as an exploration of novel RNN architectures, starting from first principles. The goal was to create a system whose memory functions more like a dynamic, resonating process rather than a static storage mechanism.

The code is the result of this experimental process. It is shared in the hope that it might provide a different perspective or spark new ideas for those interested in this field.

**(本项目由一位非专业爱好者与一个AI助手协作开发，是一次从第一性原理出发，对新型RNN架构的探索。我们的目标是创造一个其记忆功能更像一个动态共鸣过程，而非静态存储机制的系统。代码是这一实验过程的产物，我们分享它，是希望它能为该领域的兴趣者提供一个不同的视角或启发新的想法。)**

---

### **Overview (概述)**

The **Emergent Resonator** is a custom Recurrent Neural Network (RNN) architecture implemented in PyTorch.

The core concept is **"process-based memory"**. Unlike architectures that rely on an explicit memory cell (e.g., LSTM's cell state), this model represents information as a self-sustaining pattern of activity—an "information vortex"—that evolves over time based on the system's internal dynamics.

The model was tested on the sequential MNIST dataset. A key result is that a single-layer version equipped with a simple input gate achieved **~99.2% accuracy with its `reasoning_steps` (internal recurrence steps after sequence input) set to 1**.

Further experiments explored a curriculum learning strategy, progressively increasing the `reasoning_steps`. This revealed that while performance remained high, the model exhibited periods of instability (sudden loss spikes) when the cognitive depth was increased, followed by a recovery phase. This behavior is a notable characteristic of its learning dynamics under this training regime.

**(“涌现式共鸣器”是一个在PyTorch中实现的自定义RNN架构。其核心概念是“基于过程的记忆”。与依赖显式记忆单元（如LSTM的细胞状态）的架构不同，本模型将信息表征为一个自维持的活动模式——一个“信息涡旋”——它基于系统内部动力学随时间演化。该模型在序列MNIST数据集上进行了测试。一个关键结果是：一个配备了简单输入门的单层版本，在`reasoning_steps`（序列输入结束后的内部循环步数）设置为1时，取得了约`99.2%`的准确率。后续实验探索了课程学习策略，逐步增加`reasoning_steps`。实验揭示，尽管模型保持了高性能，但在认知深度增加时，它会表现出失稳（损失突然飙升）而后恢复的阶段。这种行为是其在该训练策略下一个显著的学习动态特征。)**

---

### **Core Architectural Components (核心架构组件)**

1.  **Custom Neuron Core (`PulseTGAU`):**
    *   The neuron model is inspired by spiking dynamics, incorporating concepts like membrane potential, a firing threshold, and a reset mechanism.
    *   **Intrinsic Plasticity:** The neuron's internal parameters (`threshold`, `reset_scale`, etc.) are trainable. This allows the network to learn not only the connections *between* neurons but also the fundamental response properties *of* the neurons themselves.
    *   The state update is based on additive dynamics, which helps mitigate vanishing gradients over long sequences.

    **(自定义神经元核心 (`PulseTGAU`)：该神经元模型受脉冲动态启发，包含了膜电位、发放阈值和重置机制等概念。其核心是“内在可塑性”：神经元内部参数（`threshold`, `reset_scale`等）是可训练的。这使网络不仅能学习神经元之间的连接，还能学习神经元本身的基础响应特性。其状态更新基于加性动力学，这有助于缓解长序列中的梯度消失问题。)**

2.  **Separation of Perception and Recurrence:**
    *   The model uses an input gate that is solely dependent on the current external input. This separates the initial "perception" of the signal from the subsequent "pure recurrence" phase.
    *   After the input sequence is processed, the network is allowed to evolve purely based on its internal state for a set number of `reasoning_steps`. It is in this phase that the "information vortex" stabilizes to produce a final output.

    **(感知与循环的分离：模型使用一个仅依赖于当前外部输入的输入门。这将信号的初始“感知”阶段与后续的“纯循环”阶段分离开来。在输入序列处理完毕后，网络在设定的`reasoning_steps`内，仅基于其内部状态进行演化。正是在此阶段，“信息涡旋”得以稳定并产生最终输出。)**

---

### **Intended Audience (目标受众)**

*   Researchers and students interested in RNN dynamics, alternative neural network paradigms, and dynamical systems theory in AI.
*   Developers and hobbyists exploring architectures beyond the mainstream Transformer and LSTM models.

**(对RNN动力学、替代性神经网络范式及AI中的动力学系统理论感兴趣的研究者和学生。正在探索主流Transformer和LSTM模型之外架构的开发者和爱好者。)**
