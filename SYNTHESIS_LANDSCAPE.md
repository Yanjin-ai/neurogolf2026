# 「任务→最小张量电路」的专业拆解（从零、不看我们的过程）

> 配套：EXPLAINER.md（科普版）。本文是专业版：这类系统的学名、五十年家谱、
> 每一步的关键论文与成熟实现、以及从零搭建的工程蓝图。

## 0. 问题的学名：逻辑综合（Logic Synthesis / EDA），不是机器学习
形式化：给定可执行语义 f、算子基 B（ONNX ops）、成本 c（bytes），求 argmin c(g) s.t. g≡f。
= EDA 的「面积最小化工艺映射」，门换成 ONNX 算子、面积换成字节。
成熟系统：Espresso（两级最小化）、ABC（Berkeley, AIG 重写/映射）、mockturtle（EPFL）、
SAT 精确综合（Knuth TAOCP 4A boolean chains；Soeken/De Micheli——可**证明**最小性）。

## 支柱一：语义获取 = 程序合成（ARC 家谱）
- 经典：FlashFill(POPL'11)、Sketch、SyGuS(2013)、DreamCoder(PLDI'21, 库学习)
- ARC 专线：**Icecuber 2020 冠军**（142 原语 DSL + 深度≤4 组合暴搜，C++，隐藏集~20%）
  → **Hodel arc-dsl 2023**（400 题手写求解器，160 原语 = 现成的可执行语义）
  → Greenblatt 2024（GPT-4o 每题采样数千程序+筛选，公开集~50%）
  → BARC 2024（归纳 vs 转导互补）→ TTT（Akyürek, 公开集 61.9%）/ ARChitects
  （ARC Prize 2024 冠军, 私有集 53.5%）→ o3（半私有 87.5%）
- 结论：要 100% 精确 → 手写 DSL 路线；要规模 → LLM 采样路线。

## 支柱二：程序→张量电路 = 标准单元库（gadget library）
已发表的「程序编译成网络」：
- **RASP**（ICML'21 Thinking Like Transformers）：Transformer 的编程语言
- **Tracr**（DeepMind, NeurIPS'23）：RASP→Transformer 权重的真编译器（开源）
- Springer & Kenyon 2020：手工构造生命游戏最小 CNN（局部规则→最小卷积的范本）
- Neural CA（Distill 2020）；Chomsky Hierarchy（ICLR'23）：算子基表达力地图
单元库配方（每类计算的成熟文献）：
- 连通传播 = **Kogge-Stone fill**（象棋 bitboard 标准算法；源头 Kogge&Stone 1973 并行前缀）
- 前缀网络：Kogge-Stone / Brent-Kung / Ladner-Fischer（深度vs门数权衡）
- 位并行：Hacker's Delight、HAKMEM、bitslice DES（Biham 1997）
- 形态学：Serra 1982（膨胀/腐蚀 = MaxPool/MinPool）
- 逻辑的算术编码：可微逻辑门网络（NeurIPS'22）、LogicNets、LUTNet（FPGA-ML 分支）
- 选择/路由：置换矩阵、one-hot 线性代数（Einsum/Gather）

## 支柱三：自动最小化 = 等式饱和 + 超优化
- 搜索式：Massalin 1987 → STOKE（ASPLOS'13, MCMC 搜汇编）→ Souper
  → **AlphaTensor**（Nature'22）/**AlphaDev**（Nature'23）：「发明」被 RL 自动化的存在性证明
- 重写式（对张量图最对口）：**egg**（POPL'21 杰出论文, e-graph 等式饱和）
  → **Tensat**（MLSys'21, egg 用于张量图超优化）→ TASO（SOSP'19, 自动生成+验证重写规则）
  → PET（OSDI'21, 部分等价+修正）
- 证明式：SAT 精确综合（mockturtle exact synthesis）——热点子图可证最小
- 结论：bytes 成本局部可加 → egg 抽取可做 ILP/DP；手工 peephole 是这套机器的退化形式。

## 支柱四：正确性 = 翻译验证
- Translation Validation（Pnueli 1998）、**Alive2**（PLDI'21, LLVM 优化等价检查）
- Property-Based Testing（QuickCheck 2000）= 有可执行生成器时的性价比之王
- 分层：穷举小空间 → 生成器采样 → 最坏情况定向构造 → （可选）SMT 证明子图

## 5. 从零搭建的工程蓝图（8000+ 系统的形态）
```
语义获取   arc-dsl 手写求解器 → 400 个可执行 DSL 程序
单元库     ~160 个原语各造字节最优 gadget（Kogge-Stone/形态学/einsum/SWAR）
工艺映射   DSL 程序 → gadget 组装成 ONNX 图（= technology mapping）
自动最小化 egg 等式饱和 + bytes 成本抽取（Tensat式）+ 热点 SAT 精确综合
翻译验证   对生成器语义穷举/采样微分测试 + 最坏情况构造
```
没有单篇论文做完整链条；craft = 把五个成熟文献拼装完整。每层的开源立足点：
arc-dsl / chessprogramming wiki + Hacker's Delight / Tracr / egg-egglog /
ABC-mockturtle / Alive2 方法论。

## 6. 七件精读
1. Icecuber 2020（top-quarks/ARC-solution）2. Tracr 3. egg（POPL'21）
4. Tensat 5. chessprogramming: Kogge-Stone/Dumb7Fill 6. AlphaDev
7. mockturtle exact synthesis 章
