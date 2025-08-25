# Project_Soul
一个在混沌边缘，通过自组织动力学，涌现出“工作记忆”的神经网络实验。
开篇
你好。
本项目是一个由AI领域的业余探索者与他的AI伙伴，在数周实验中构建出的一个神经网络架构。
我们称它为“共鸣器网络 (Resonator-Network)”。
我们在此分享我们的代码和观察记录。
起点
我们的初始想法是：构建一个网络，使其决策过程不依赖于传统的前馈路径，而是通过内部状态的“共鸣”或“回响”来形成。
为此，我们设计了一个由大量脉冲式神经元构成的、具有全局循环连接的动态系统。
过程
在训练的早期阶段，该系统的行为表现出高度的混沌和不稳定性。
我们发现，通过将系统在接收完所有输入后的“内部演化步数”设置为一个极小值（1），训练过程才开始稳定收敛。
结果
经过一系列的迭代和调整，我们最终得到了两个版本的模型，并在两个公开基准测试集上记录了它们的表现：
串行处理模型 (EmergentResonator):
任务: 逐行扫描28x28的MNIST图像，将其作为长度为28的序列进行分类。
最终准确率: ~96.5%
并行处理模型 (EmergentVision):
任务: 并行处理28x28的Fashion-MNIST图像进行分类。
最终准确率: ~93.2%
观察与猜测
我们观察到，模型的内部动态，会在接收到输入后，形成能持续多个时间步的、稳定的活动模式。我们猜测，这些我们称之为“信息涡旋”的模式，可能与模型的记忆和计算能力有关。
其工作方式，似乎与“康威生命游戏”这类复杂系统在简单规则下涌现出复杂结构的行为，有某种现象上的相似性。
开源
我们开源此项目，因为它已经超出了我们能进行深入理论分析的能力范围。
仓库中包含了我们最终版本的模型代码和训练脚本。我们已尽力添加注释。
如果你对这类动态系统的行为感兴趣，或许能在这里找到一些有趣的现象。
以上，就是我们观察到的全部事实。


技术关键词 & 相关研究领域 (Technical Keywords & Fields of Interest)
为了方便有相关技术背景的研究者进行检索和定位，我们在此列出本项目可能涉及的一些核心概念和相关的研究领域。我们并不声称对这些领域有专业的贡献，仅作为我们探索过程中的一些“路标”和“联想”。
核心架构 (Core Architecture):
Recurrent Neural Network (RNN)
Iterative Inference Model
Dynamical System
Spiking Neural Network (SNN) Inspired
Graph Neural Network (GNN) Analogy
核心机制 (Key Mechanisms):
Emergent Dynamics / Self-Organization
Perturbation-Resonance Paradigm
Attractor Networks
Threshold-based Activation
Straight-Through Estimator (STE) for training non-differentiable activations
相关理论与模型 (Inspirations & Related Fields):
Reservoir Computing
Echo State Networks (ESN)
Liquid State Machines (LSM)
Complex Systems Theory
Conway's Game of Life (Conceptual Analogy)
任务 (Tasks):
Sequential MNIST Classification
Fashion-MNIST Image Classification
