# NeuroGolf 项目通俗深讲：它是什么、属于什么体系、如何连接 AI 前沿

> 配套 RETROSPECTIVE.md（复盘）与 GRAPH_COMPILER_CURRICULUM.md（技能课程）。
> 本文回答四个问题：要实现什么 / 属于哪个研究体系 / 论文与开源对应物 /
> 如何赋能当下 AI + 硬技术清单。

## 一、要实现什么（比喻版）
普通 AI 比赛 = 教学生做题；NeuroGolf = 为每道题造一台**最小的专用机器**。
400 道 ARC 图形题，每题一张 ONNX 计算图，必须 100% 精确，得分 = 25 − ln(体积)，
体积 = 参数 + 中间结果字节。本质是"计算的极限压缩"：代码高尔夫的神经网络版、
钟表匠、demoscene。实例：task002 封闭区域填充，笨法几十个 900 字节网格模拟水漫，
专家把一行 30 格打包进一个整数用位移+OR 并行漫延，120 字节收工。

## 二、属于哪个研究体系（五个交叉点）
1. **程序合成**：ARC = Chollet 2019 "On the Measure of Intelligence"，从极少样例
   抽象规则，AGI 评测主赛道，至今未解。
2. **ML 编译器**：我们的 cost-golf = 手工版编译器优化 pass（融合/折叠/降级/消除），
   工业对应 TVM/MLIR/XLA/torch.compile，只是目标是延迟不是字节。
3. **超优化**：STOKE、AlphaTensor(Nature'22)、AlphaDev(Nature'23)——"语义固定、
   成本最小化"的搜索。NeuroGolf 就是这样一个搜索空间；头部 8060 = 更深的搜索。
4. **机制可解释性的镜像**：interp 把网络逆向成算法；我们把算法正向编译成最小网络。
   逆向专家电路的方法（op直方图→张量成本→einsum方程→producer链）与拆
   Transformer 同源。
5. **电路复杂度**：成本地板 = "实现该函数最少需要多少门"的实践版。

## 三、论文与开源对应物
- ARC：Chollet 2019；Hodel arc-dsl（本仓库 third_party）；DreamCoder；
  ARC Prize 2024/25 报告（test-time training + 程序合成）。
- 图优化：TASO(SOSP'19)、PET(OSDI'21)（自动等价重写）；onnx/optimizer、
  ORT graph optimizations、onnx-simplifier/onnxslim（实测 +0.023 的教训：
  机械 pass 打不过专家手工）；TVM/MLIR/Triton。
- 算法发现：AlphaTensor、AlphaDev——"NeuroGolf 头部靠什么"的理论答案。
- 底层：《Hacker's Delight》（SWAR 圣经）、象棋 bitboard。
- 工业同构：**FlashAttention**——见下。

## 四、如何赋能当下 AI
1. **LLM 推理成本 = 同一个数学对象**：GPU 瓶颈是内存搬运不是算力。
   FlashAttention 快 3 倍的原因 = 不物化 N×N 中间矩阵 = 专家电路不物化 30 个
   中间张量。我们实测的"瓶颈迁移律"（大张量→dtype→张量个数）在工业界叫
   kernel fusion / 量化 / 激活重计算。
2. **AI 编译器兵家必争**：torch.compile、Triton、自研芯片编译栈。
3. **AI 发明算法是下一前沿**："+27.3 中 0 分来自创造"反过来读：谁自动化了
   "创造"，谁拿走剩下 790 分——AI-for-AI-efficiency 的研究命题。
4. **可验证 AI**：三重验证门 + 来源可疑度分级（task192/merge5 两次血案）
   = "如何信任计算产物"的微缩实践。

## 五、硬技术清单（四层）
- **数学/算法**：einsum 张量代数（地基）、SWAR 位并行、scan/reduce/argmax
  并行原语、形态学（膨胀=MaxPool）、复杂度下界直觉。
- **表示/规范**：ONNX 算子语义与 SSA 图、数值格式陷阱（float 整数精度上限
  2^24，实撞）、算子×dtype 内核支持矩阵（CumSum 无 int8、Einsum 无 uint8，
  全靠实测）。
- **编译器技术**：融合、折叠、死代码消除、强度削减（算术顶替分支）、
  成本模型驱动的 rewrite 搜索。
- **实验工程**：电路逆向工作流、验证科学（跟踪误差 0.4 标定、fresh 门、
  来源分级）、可复现基础设施（子进程隔离/磁盘卫生/floor-safe 设计）。

**收拢**：给定功能求最小计算图 = 程序合成 × 编译器 × 超优化的交点；其成本
模型与 FlashAttention/量化/融合是同一个数学对象；练出的技能就是 AI infra
与 efficiency 研究的看家本领；AlphaTensor/AlphaDev 指明未来——把"发明"
变成搜索交给 AI。
