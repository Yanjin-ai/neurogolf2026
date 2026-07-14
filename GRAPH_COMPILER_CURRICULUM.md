# 图级编译器技能线：2-4 周训练课程

> 从 NeuroGolf 2026 的实战诊断演化出的长期能力建设方案。
> 目标不是比赛分数，是把「图级张量代数 / 微型编译器设计」建成可迁移技能，
> 服务于 thesis / LLM graph-schedule / compiler-agent 方向。
>
> **本课程全部教材来自本仓库的实物资产**——不是抽象教程，是解剖真实的专家作品。

## 教材清单（已在仓库中）

| 资产 | 路径 | 用途 |
|---|---|---|
| 专家紧电路（68节点全注释解读） | `COST_GOLF_PLAYBOOK.md` B11 + `merge_work/2026-neurogolf-baseline/task064.onnx`* | 精读教材 |
| 我的 v1→v4 失败曲线（259642→45259 vs 专家10358） | `tools/build_task064_v{1..4}.py` + B10 | 反面教材：张量个数的坎 |
| 验证过的参考实现（266+3000全对，未移植） | `tools/task054_ref.py` | 现成的期末练习题 |
| 平台硬边界手册（实测） | B4/B10：CumSum只int32、Einsum不支持uint8、OneHot规则 | 设计约束速查 |
| 电路逆向工作流 | B8 + `tools/reachable_scan.py`、`technique_mine.py` | 解剖工具 |
| 评分/验证管线 | `tools/lean_score.py`、`src/validate.py`（fresh 门槛） | 每个练习的验收器 |

*bundle 不在 git 里，本地有；或从 Kaggle 重新拉。

---

## 第 1 周：脑内 DSL——把逻辑想成张量方程

**目标状态**：看到 `'bchw,ec->beh'` 直接读出「选颜色+投影掉W→行剖面」，不查文档。

1. **Einsum 私塾**（2-3 天）：用 `np.einsum` 做 20 个小练习——1D/2D 投影、外积、
   缩并、批量选择。每个先写自然语言语义，再写方程，再验证。
2. **精读 task64 专家电路**（2-3 天）：对 68 个节点逐个写出
   （a）张量方程 （b）几何意义 （c）为什么这个 dtype/形状。
   验收：不看我的 B11 注解，独立重建出 5 个技术的解读。
3. **练习**：给 3 个简单变换（选主色填充/画边框/对角镜像）各写"纸上电路"
   ——只允许 Einsum/Reduce/ArgMax/Where/OneHot，中间张量 ≤8 个。

## 第 2 周：重算子当指令用——ArgMax/TopK/OneHot/Where

**目标状态**：离散决策（选色/找边界/条件填充）条件反射地映射到重算子，
而不是布尔张量链。

1. **重做 task64**：从零建 v5，规则：全图中间张量 ≤12 个（专家是 ~5 个 fullT）。
   工具链：`build_task064_v4.py` 起步 + B11 五技术。验收：`lean_score` ≤ 15000
   （不要求打进 10358——要求进入同一个数量级）。
2. **平台边界测试驱动**：写一个 `tools/op_support_probe.py`，系统探测
   每个候选算子 × dtype 的 ORT 支持矩阵（B4 只有零散实测），沉淀成表。
3. **练习**：把 `task054_ref.py` 的「颜色角色检测」段（bg/box/c0/c1/c2/vert/horiz）
   单独移植成 ONNX 子图，目标 ≤10 个中间张量、全部非 float。

## 第 3 周：SWAR / bit-packing——压缩状态机

**目标状态**：能把 2D 布尔传播写成位打包整数上的移位-OR 链，并证明收敛步数。

1. **解剖现成 SWAR 电路**：`merge_work/*/task002.onnx`（bit-packed flood-fill，
   BitShift×27+BitwiseOr×47）。用 B8 工作流逐节点解读打包/传播/解包三段。
2. **numpy SWAR 练习**：写 2D→packed→传播→解码 纯函数，处理三个坑
   （float 2^24 精度上限、跨行泄漏 mask、最坏尺寸收敛）。
3. **ONNX 映射**：把上面的链映射成 ONNX（Cast/Mul/BitShift/BitwiseOr），
   在 task002 上验证正确（不要求成本赢——现网络已是地板）。

## 第 4 周：期末——task054 完整移植 + 复盘

1. **移植 `task054_ref.py` → ONNX**：参考实现已 266+3000 全对，规则零风险。
   我的预算核算是 ~60k（B10 口径），专家线 25144。目标分级：
   - 及格：正确 + ≤50k（打败我自己的预算）
   - 良好：≤35k
   - 优秀：≤25144（打破公开地板——如果做到，这题值 +0.6 分且证明技能成型）
2. **写复盘**：v1→vN 曲线 + 每刀省在哪 + 还差什么。对照 task64 的 B10 曲线，
   量化四周训练带来的「张量个数直觉」改善。
3. **展望映射**：写一页纸，把学到的图级思维映射到目标研究方向
   （zero-BP 的状态周转最小化 / LLM graph-schedule / compiler-agent 的
   rewrite-rule 搜索）——每个方向找出一个直接可复用的模式。

---

## 工具与文献（开源对应物）

- **onnx/optimizer**、**ONNX Runtime graph optimizations**（Basic/Extended/Layout
  pass）、HF **Optimum** O1-O4：主流图优化栈，优化目标是延迟/内存而非 bytes-cost，
  但 pass 设计（常量折叠/死节点消除/融合）直接可借鉴到 cost-golf 优化器。
  已有雏形：`tools/surgery_pipeline.py`（复刻的社区手术目录，实测 +0.023 的
  教训=机械 pass 打不过专家手工，见 B10）。
- **TVM / MLIR**：rewrite-rule + 搜索的正统框架；第 4 周展望时对照。
- **ARC-AGI 文献**（NVARC 等）：结构化任务分解 + synthetic data + test-time
  training；与本课程互补（他们解决"理解"，本课程解决"编码成最小电路"）。

## 与比赛的关系（诚实的边界）

本课程**不承诺**打破 NeuroGolf 公开地板。诊断（B10/B11 + task54 预算核算）表明：
打进专家张量数需要的图级融合直觉，是数周-数月的刻意练习，不是临场可补。
课程的验收标准是**自己的曲线改善**（v1→vN 的张量数/成本下降斜率），
不是 leaderboard。若第 4 周 task054 真打进 25144，那是 bonus，不是要求。
