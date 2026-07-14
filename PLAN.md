# NeuroGolf 2026 冲前10作战计划

> 更新：2026-07-07。截止：**2026-07-15 23:59 UTC（剩 8 天）**。
> 榜首 8060.17 / 前10门槛 7919.70（会继续上涨，按 **8000–8100** 规划目标）。
> 我方现状：最强公开 baseline（LB 7243.33）已到手并本地复评 = **7169.77**（4 题本地失败待查）。
> 总分理论上限 10000；baseline 剩余提升空间（headroom）= **2830 分**。

---

## 0. 已验证的事实基础（全部实测/官方代码确认，勿再怀疑）

### 赛制
- 400 题 = **ARC-AGI-1 公开训练集**，taskNNN = 按 hex 排序的第 N 个任务（task001 = `007bbfb7`）。
- 每题 JSON：`train`（原始训练对，2–10 个）+ `test`（原始测试对，~1 个）+ `arc-gen`（~250 个官方生成样例）。
- 判分：网络必须在 **全部 train+test+arc-gen + 私有小样例集** 上逐像素全对，才拿分；否则该题 0 分。>30×30 的样例被跳过。
- 每题得分 `max(1, 25 − ln(max(1, memory + params)))`，总分为 400 题之和。
- 提交：`submission.zip` 内 `task001.onnx … task400.onnx`，每题至多一个文件，缺失记 0。
- 限额：规则写 **5 次/天**（API 显示 100/天，需实测），最终可选 2 个提交。
- 评分环境：**python 3.12 / numpy 2.4.4 / onnx 1.21.0 / onnxruntime 1.24.4 / onnx-tool 1.0.1**，`ORT_DISABLE_ALL`（不做图优化），先 sanitize（重命名全部张量为 safe_name_N）。
- 奖金：$12k/$10k/$10k + 最佳学生队 $8k + 最长霸榜 $10k。主办方保留改评分规则并重评的权利（已有人证明仍存在 25 分漏洞，**评分规则中途再变是真实风险**）。

### Cost 模型（= 一切设计决策的物理定律）
- `params` = 所有 initializer + Constant 节点的**元素个数**（与 dtype 无关；标量=1；**多节点共享同一 initializer 只计一次**）。
- `memory` = 所有**中间张量**的字节数（静态 shape × dtype 字节宽，取 profiler 实测最大值）。**名为 `input`/`output` 的张量免费；节点属性免费。**
- 推论（实测成本表，本地已复现）：

| 电路模式 | cost | 得分 |
|---|---|---|
| 单节点纯属性算子（Transpose / Upsample-7 / Slice-1 / Pad-2 / Pool…） | 0 | **25.0** |
| 单节点 + 标量 initializer | 1 | **25.0** |
| 镜像（Slice steps=−1，4 个 int64） | 4 | 23.6 |
| 全局条件选择（ReduceSum→Greater→Where） | 6 | 23.2 |
| 颜色重映射（单 Gather axis=1，10 个 idx） | 10 | 22.7 |
| 行/列平移（单 Gather，30 idx） | 30 | 21.6 |
| 颜色计数比较 + Where | ~54 | 21.0 |
| 众数颜色填充（reduce→argmax→equal→Expand） | ~116 | 20.3 |
| 单层 3×3 Conv（10×10×3×3） | 900 | 18.2 |
| rot90（单 Einsum + 30×30 置换阵） | 900 | 18.2 |
| WTA 重编码（ReduceMax→GreaterOrEqual 作为末节点） | 3600 | 16.8 |
| 3× 放大（Slice-1 裁剪→Upsample-7 末节点） | 4000 | 16.7 |
| bbox 掩码（行×列占用） | ~7400 | 16.1 |
| 任意逐像素置换（GatherElements 末节点，9000 int idx） | 9000 | 15.9 |
| 一个 fp32 全尺寸中间张量 [1,10,30,30] | ≥36000 | ≤14.5 |
| 洪泛/连通域（30–60 步展开 fp16 Conv+Min，共享权重） | 117k–225k | 13.3–12.7 |

- **设计定律**：
  1. 1 param = 1 byte = 1 cost。int 常量放 initializer（每元素 1），不要算出来（int64 中间张量每元素 8）。
  2. 首节点读 input 免费、末节点写 output 免费 → **“早收缩、晚展开”**：大张量只出现在首末，内部只留极小张量（reduce 到 [1,10,1,1]=40B 后逻辑几乎免费）。
  3. 中间张量 dtype：bool 1B / fp16 2B / fp32 4B / int64 8B。
  4. 用**老 opset 的属性版算子**（每个 .onnx 可自选 opset）：Slice-1、Pad-2、Upsample-7、Split-2、带 axes 属性的 Reduce* —— 全零参数。注意 Equal(float) 需 opset≥11、GreaterOrEqual 需 ≥12，按需混合选择单题 opset。
  5. `Where` 合法（只有 If 被禁）；条件逻辑在 reduce 后的小张量上做。
  6. 输出只需**符号正确**（>0 视为 1，全 ≤0 视为 padding）→ 不需要精确 one-hot，可用 hinge 松弛训练后固化。
  7. 输出网格尺寸由“尾部全零行列”隐式决定 → 变尺寸输出用行/列掩码静态实现，无需动态 shape。
  8. sparse_initializer 有官方 sanitize bug（重命名悬空引用会崩），**禁用**。
  9. 文件 1.44MB 上限几乎不构成约束（900 参数 Conv ≈ 3.7KB）。
  10. 用 `onnx.helper` 直接建图（ir_version=10），**不走 PyTorch 导出**（会喷 Cast/Constant/Reshape 垃圾）。

### 弹药库
- **google/arc-gen**（已克隆 `third_party/arc-gen`）：每题一个可读 Python 生成器 = **真实规则即源码**，且是官方私有样例的同分布生成器 → `arc_gen.py generate <id> N` 可无限生成“私有分布”验证集。**任何网络上线前必须过万级生成样例。**
- **michaelhodel/arc-dsl**（已克隆 `third_party/arc-dsl`）：400 题符号 solver。实测：**280/400 题在全部官方样例上通过**（可直接当可执行真值/编译源）；120 题在 arc-gen 分布上失败（只拟合了原始样例，须以 arc-gen 生成器为真值）。清单：`experiments/arcdsl_vs_arcgen.json`。
- **baseline bundle**（`baselines/udit/nets/`，LB 7243.33）：本地复评 7169.77，逐题成本表 `baselines/udit/scores.json`、优先级表 `experiments/priority.json`。
- 社区已公开的技法：自门控 Einsum 恒等（`nchw,nkwh->nchw`，0 cost）、SGD 拟合符号模式小网络、uint8 形态学流水线、graph surgery 清理栈（seddiktrk）。
- 已知教训：**换入未验证的公开便宜网络 → 私有集 0 分**（实例：预期 +116 实际 −308）。手工实现真实规则的网络与官方评分误差 <0.01。

### Baseline 逐题分布（我们的起点）
- 25 分：3 题 ｜ 22–25：9 题 ｜ 19–22：97 题 ｜ **16–19：248 题** ｜ 13–16：39 题 ｜ 0 分：4 题（task220/230/294/352，本地失败待查——可能是 macOS ORT 行为差异或该网络本身脆弱，无论哪种都要替换成自建网络）。
- 达到 8050 ≈ 平均每题 20.1 分 ≈ 平均 cost 128。当前中位数 17.8（cost ~1300）。**核心工作 = 把 16–19 分段的 248 题批量抬升 2–6 分**，头部 40 个最差题（cost 9k–34k）单题就有 6–9 分可挖。

---

## 1. 总体判断

这不是推理比赛，是**已知规则的电路编译 + 成本压缩**比赛：
- 任务归因层不需要“猜规则”——规则在 arc-gen 源码里；arc-dsl 给了 280 题现成可执行程序。
- 你原计划的六层架构收缩为四件实事：**真值获取（读源码/跑 solver）→ 模板化 ONNX 编译 → 逐题成本压缩 → 生成式抗过拟合验证**。
- 8 天内的胜负手：单位时间产出的“分/题”。log 尺度意味着**重设计 >> 微调**（cost 砍半只 +0.7 分，换架构砍 10 倍 +2.3 分）。

## 2. 系统架构（工程版，已部分就位）

```
repo/
├── data/                      # 官方数据（已就位，400 题 + neurogolf_utils.py）
├── third_party/arc-gen/       # 真值规则源码 + 私有分布生成器（已克隆）
├── third_party/arc-dsl/       # 400 题符号 solver（已克隆）
├── baselines/udit/            # LB 7243 公开 bundle + 逐题评分（已就位）
├── src/
│   ├── oracle.py              # 官方评分复刻（已就位，误差=0）
│   ├── graphlib.py            # onnx.helper 建图工具：节点/initializer/opset 选择/value_info 自动补全
│   ├── templates/             # 标准单元库（§3 模板族，每族一个构造函数 + 成本估计）
│   ├── compile/               # 每题一个 build_taskNNN() —— 人/半自动写，产出候选网络
│   ├── surgery.py             # 后优化：initializer 去重/共享、int64→int32、去默认输入、fp16/bool 降精度（逐题验证收益）
│   └── validate.py            # 三级验证：官方样例全过 → arc-gen 现生成 2k–10k 样例全过 → cost 复核
├── tools/
│   ├── priority.py            # 按 (25−当前分) × 预估可行性 排序的工作队列
│   ├── gen_examples.py        # 调 arc-gen 批量生成验证样例（缓存到 gen_cache/）
│   └── package.py             # 择优合并（每题取已验证候选中 cost 最低者）→ submission.zip
└── experiments/               # 成本实验、priority.json、arcdsl_vs_arcgen.json
```

**择优合并原则**：每题维护候选池 {baseline 网络, 我方网络…}，只有通过三级验证的候选才可参与，取 cost 最低者打包。任何时刻都能打出一份不低于 baseline 的合法提交。

## 3. 模板库（按设计阶梯排列，对应实测成本表）

| 阶梯 | 模板族 | 目标分 | 适用 |
|---|---|---|---|
| L0 | 单节点属性算子：Transpose/Slice-1/Pad-2/Upsample-7/Pool/DepthToSpace/自门控 Einsum | 25 | 几何变换、裁剪、平铺、缩放、恒等类 |
| L1 | 单节点小 initializer：Gather 换色(10)、Gather 移位(30)、镜像 Slice(4)、小稀疏 Conv | 21.6–25 | 颜色映射、平移、局部模式 |
| L2 | 标量域逻辑：Reduce 到 [1,10,1,1] → Greater/Equal/Where → 末节点 Expand/广播回写 | 19–23 | 计数、比色、全局条件、选择性重绘 |
| L3 | 单 Conv / 单 Einsum / GatherElements 末节点 | 15.9–18.2 | 局部规则、rot90、任意像素置换 |
| L4 | bool/fp16 网格域多节点：形态学、bbox、射线扫描（CumSum）、WTA 末节点 | 14.5–16.8 | 区域填充、包围检测、对象几何 |
| L5 | 展开迭代（共享权重 Conv+Min/MaxPool 传播，步数按任务实际最大网格裁剪） | 12.7–14 | 真·连通域/洪泛，最后手段 |

模板选择原则：**先问”能不能一个节点”，再问”能不能只在小张量上算”，最后才进网格域。** L5 的迭代步数不要按 60 的最坏情况，按该题 arc-gen 分布实测最大传播距离 + 余量。

### 3.5 实战验证的原语手册（随开发滚动补充）

- **RoiAlign 万能空间变换**（学自 baseline task087）：单节点 5 参数（rois[1,4]+batch_indices[1]），负坐标 ROI 可实现固定裁剪/翻转/rot180/缩放采样，attributes 里 output_h/w 免费。
- **一热恒等式**：任意在格单元 Σ_c input[c,r,w]=1、padding=0 → “在格指示器”是输入的线性函数（fix4 的 padding 抑制、ReduceL2 尺寸提取都靠它）。
- **√ReduceL2 提取正方形边长**（学自 baseline task150）：ReduceL2 全张量 = √(h·w)，正方形网格即边长，标量域 4 字节。
- **Range+Gather 负索引回卷**：Gather 负 idx 自动取尾部（padding 全零列），变尺寸镜像 cost≈138。
- **Einsum 选择矩阵**（学自 baseline task380）：小固定网格的行/列线性变换（旋转/置换），(30,s)+(s,s) 选择器单节点。
- **自门控 Einsum**（社区公开）：`nchw,nkwh->nchw` 等二次型恒等，0 cost 满分。
- **整数权重纪律**：所有手工电路用整数权重、双侧 margin ≥1，禁止贴 0 边缘（baseline 有网络因此在本机翻车）。
- **ORT 1.24.4 双坑**：group Conv 的 bias/filter 行按组平铺（第二组复用第一组）——权重设计须在规范/平铺两种语义下都符号正确；同进程内先建的 session 会污染后续同构模型 → 一切批量评测走子进程（batch_score.py / validate_isolated）。
- **提交格式**：zip 文件名必须严格叫 `submission.zip`。

## 4. 每日推进计划（剩 8 天）

**D0（今天 7-07）——地基收尾 + 首次提交**
- [x] 环境对齐官方（onnx 1.21/ort 1.24.4/numpy 2.4.4）
- [x] 官方评分 oracle 复刻、baseline 逐题成本表、arc-dsl 280/120 分割、优先级表
- [ ] `graphlib.py` + `validate.py`（含 arc-gen 生成器调用）+ `package.py`
- [ ] 查清 4 个本地失败题（220/230/294/352），用自建网络替换
- [ ] **原样提交 baseline bundle 确认分数与流程**（也实测每日限额到底是 5 还是 100）

**D1（7-08）——模板库 + 高分段扫尾**
- 实现 L0–L2 模板族；写自动匹配器：对 400 题逐一尝试 L0/L1 模板（几何族、换色族、移位族已有 11 题确认，社区 Einsum 门控等再扫一轮）
- 22–25 分段和 19–22 分段里挑“规则简单但 baseline 网络臃肿”的题批量重写（预期 +50–150 分）

**D2–D3（7-09~10）——主攻 40 个最差题（cost 9k–34k，每题可挖 6–9 分）**
- 流程固定：读 `third_party/arc-gen/tasks/<id>.py` 源码（真实规则）→ 选阶梯模板 → 手写 build_taskNNN() → 三级验证 → 入候选池
- dsl_ok=True 的题可先跑 arc-dsl solver 理解结构；dsl_ok=False 的 120 题只信 arc-gen 源码
- 每天固定提交一次，跟踪 LB 校准（本地分 vs 线上分的逐题残差）
- 预期：40 题 × 平均 +5 ≈ **+200 分**

**D4–D5（7-11~12）——批量抬升 16–19 分段（248 题）**
- 按 priority.json 从 cost 高到低推；每题限时 20–40 分钟，超时跳过（log 尺度下不恋战）
- 半自动化：对 arc-dsl solver ≤5 行的 ~80 题，写 primitive→模板 的映射器辅助生成初稿
- 通用 surgery pass 对全部候选跑一遍（去重/共享/降精度/int32 索引），每题验证后择优
- 预期：150–200 题 × 平均 +2 ≈ **+300–400 分**

**D6（7-13）——第二轮迭代 + 风险对冲**
- 复盘 LB 残差，修正本地 oracle 与线上的任何偏差
- 对私有集风险最高的题（规则里有低概率分支的）加大生成样例量到 10k+ 重验
- 若评分规则被官方调整：模板库按新规则重排序（我们的网络都是“干净”实现，抗规则变化）

**D7（7-14~15）——冻结与收尾**
- 最后一轮择优打包；提交 2 个最终候选：①最高分版本 ②去掉私有集风险最高改动的保守版本
- 留出 24h 缓冲应对 Kaggle 排队/重评

**产出预算**：7243（地板）+ 70（修 4 题）+ 100（L0–L2 扫尾）+ 200（最差 40 题）+ 350（中段批量）≈ **7960–8060**，即前 10 边缘到前 5。上不封顶的部分来自 D6 的第二轮和自动化程度。

## 4.5 评审吸收（2026-07-07 复盘确认的增强项）

- **自动化优先级上调**：D4 前先花半天写 arc-dsl primitive→模板 的 translator（覆盖 ≤5 行 solver 的 ~80 题），用 0.5–1 天换后期批量产能。
- **tiny weight fitter 作为 L3/L4 兜底**：结构定了但权重难手写时，利用 >0 符号松弛做 hinge 拟合（10–100 参数小 conv），自动化产出。
- **validate.py 增加扰动测试**：对 arc-gen 生成输入做颜色标签置换/平移/加噪版变体（在任务不变性允许范围内），识别"记忆映射"型脆弱网络。
- **按任务家族分批处理**：同族题（换色族/镜像平移族/计数族…）集中改写，减少认知切换。
- **首次提交时线上复核 sign-based 判据**（本地官方代码已确认 `(out>0)` 判分，仅做交叉确认）。

## 5. 验证纪律（防私有集清零，一票否决）

1. **官方样例**：train+test+arc-gen 全对（oracle.py，与线上评分同栈）。
2. **生成式验证**：arc-gen 生成器现场生成 ≥2000 个新样例全对（可疑题加到 10k）；这是私有集的同分布模拟，**不过此关不准入池**。
3. **成本复核**：sanitize 后重测 memory+params；确认无禁用 op、无 GRAPH 属性、单输入单输出、静态 shape、value_info 完整、文件 <1.44MB。
4. 最终包在 Kaggle notebook 里用官方 `verify_network` 抽查若干题（对冲 macOS/Linux ORT 行为差异——本地 4 题失败事件已证明这个风险真实存在）。

## 6. 风险登记

| 风险 | 概率 | 对策 |
|---|---|---|
| 评分规则中途再改（已有未修复漏洞的公开讨论） | 中 | 只做“干净”实现；文件字节数顺手控制（若切到 file-size 计分我们仍有序） |
| 私有集清零 | 中 | 验证纪律 §5；两个最终提交一激进一保守 |
| 本地 ORT(macOS) 与线上(Linux) 行为差异 | 已发生 | 关键题在 Kaggle 环境复验；避免 ORT 边缘行为（负 pad Conv 等） |
| 每日提交限额只有 5 次 | 中 | 本地 oracle 当唯一日常裁判，LB 只做校准 |
| 时间不够覆盖全部 248 题 | 高 | 严格按 headroom×可行性排序，单题限时，log 尺度下永远先做最差的题 |

## 7. 与原六层规划的对应

| 原规划 | 落地形态 |
|---|---|
| 任务表示层 | 不需要重建——arc-gen 源码 + arc-dsl solver 即结构化真值 |
| 任务归因层 | 退化为“读源码 + 模板匹配器”（priority.py + 自动 L0/L1 扫描） |
| DSL/IR 层 | arc-dsl 的 160 primitive 即现成 IR；我们只写 primitive→模板映射 |
| 网络模板库 | §3 六级设计阶梯（实测成本定价） |
| 合成器 | 模板编译为主 + 少量 SGD 符号拟合（利用 >0 松弛）兜底 |
| 成本优化层 | surgery.py 通用 pass + 逐题重设计（重设计优先） |
| 验证编排 | §5 三级验证 + 择优打包 package.py |
