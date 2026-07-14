# NeuroGolf Cost-Golf Playbook + 心路全背景

> 写于 2026-07-13。本文两部分：**A. 我怎么一步步走到"读现网络电路学技术"这个方向**（全背景，含所有弯路和元错误）；**B. ONNX/张量代数 cost-golf 技术手册**（实证于 task64 现网络电路，非空谈）。

---

# A. 全背景：这个方向是怎么逼出来的

## A0. 起点与重定标
- 比赛 NeuroGolf 2026：把 400 个 ARC 任务各编译成一个最小 ONNX 网络。`cost = 参数元素数 + 中间激活字节数`；`points = 25 − ln(cost)`；input/output 张量和节点属性**免费**。
- 用户接手时 baseline = 7243（rank 889 = top30%）。精确门槛：top10%=7271、top5%=7305、top2%=7393。

## A1. 第一条路：跨公开 bundle 合并（成功但触顶）
- 下载 18 个公开提交，`tools/merge_bundles.py` 逐题取「最便宜且正确」的网络。
- **merge2 = 7268，Kaggle 确认（本地 7268.47 vs Kaggle 7268.05，误差 0.42）→ 本地评分器保真。从 top30% 跳到 top10%。**
- 但**合并本质是"选择"不是"创造"**，硬上限 = 公开解逐题并集 = 7268。

## A2. 四条机械路径，全部独立触顶 7268
| 路径 | 本质 | 结果 |
|---|---|---|
| 合并 | 整网选择 | 7268 |
| 图手术（复刻 seddiktrk 全套 FP16/int8/index/broadcast，`tools/surgery_pipeline.py`） | 逐网机械施加技术 | **+0.023** |
| 局部拟合（`tools/attack_local.py`，82 局部题 LP 单Conv） | 统计拟合 | **0 wins** |
| 技术迁移自动检测（`tools/technique_mine.py`，按生成器结构聚类） | 找同结构成本离群 | **0（6/6 假阳性）** |

**关键教训**：手术只证明**编码(dtype)到顶**，非**算法到顶**。排行榜有人 8060（比 7268 高 792）→ 算法层有真余量。

## A3. 元错误：我反复"从理论宣布墙"
- 我多次断言"昂贵题已贴成本地板""einsum 融合是不可学的专家跳跃"——**但都是从理论下结论，没读过现网络电路**。
- 用户每次逼我往下走，往下就都还有东西。这是本 session 最大的元教训：**遇到卡点要去逆向拆解现网络，不要从理论宣布墙。**

## A4. 手工攻坚 task64（2c608aff）——第一次端到端跑通
- 规则（读生成器 `third_party/arc-gen/tasks/task_2c608aff.py`）：实心矩形 box + 散点 dot；每个与 box **正交对齐**的 dot 向 box 射同色射线；对角 dot 不动。
- 我手建电路（`tools/build_task064.py`、`_v2.py`），**正确 267/0**，成本一路 golf：259642 → 162442(Gather) → 84876(极值法) → 56886(int8)。
- 但仍 5.4× 于现网络 10358。**兜底锁死：`protected/task064_correct_v1_cost162442.onnx`（只读）。**

## A5. 转折：真读现网络电路 → 发现技术是可学的
现网络 task64（10358）用了三个我没用的具体技术（见 B 部分实证）：
1. **Slice 到实际尺寸 24×24**（我全程 30×30，白多 36%）
2. **Einsum 融合的 1D 投影 + 抬升**（我物化了 ~30 个 2D 张量）
3. **TopK/OneHot 紧凑选色**（我用 ArgMax 长链）

**结论修正**：所谓"墙"不是天赋，是**三个可学、可迁移的技术**。真正的技术库不在生成器里（A2 挖错地方），**在现网络的电路里**。方向 = **逆向拆解现网络 → 提取技术 → 重构**。

---

# B. Cost-Golf 技术手册（实证于 task64 现网络）

## B0. 成本模型的三条第一性原理
1. **只有中间张量和参数计费**。input/output 张量、节点属性（Constant 的 value、arange 常量）**免费**。
   → 推论：能塞进"图输出"或"属性/常量"的，不要做成中间张量。
2. **成本 = Σ(张量元素数 × dtype字节)**。三个乘数都要压：**元素数、张量个数、dtype**。
3. **一个算子 = 一个输出张量，但算子内部可以做任意多计算**。
   → 推论：**用"重"算子（Einsum/Conv/Reduce/CumSum/TopK）把多步逻辑折进一个张量**，而不是拆成几十个小算子各自物化。

## B1. 【核心技术】Einsum 融合投影 —— 可分离化的引擎
task64 现网络的命门，实证方程：
```
rowproj = Einsum('bchw,ec->beh', input, cand_oh)   # 选色(缩并c) + 投影掉W → 每行 [b,色,H]
colproj = Einsum('bchw,ec->bew', input, cand_oh)   # 选色 + 投影掉H → 每列 [b,色,W]
M       = Einsum('bchw,ec->behw', input, twohot)   # 只选/重组通道 → 工作区 [b,选,H,W]
```
**一个 Einsum 同时做两件事：① 沿 channel 轴缩并 = 选颜色（代替 Gather+mask 物化）；② 丢掉一个空间轴 = 投影到 1D。**
- `bchw,ec->beh`：把 30×30×10 压成 30（每行），**900→30，30×压缩**。
- 我的错误对照：我用 `Gather+Unsqueeze+Cast` 物化 [1,1,30,30] mask（3 张量/色），再在 2D 上算极值。应该：`Einsum('bchw,ec->beh')` 直接拿 1D 行剖面。

**可分离化判据**：如果 output(h,w) 能写成 `f(行信息 h) 和 g(列信息 w) 的组合`，就是可分离的 → 在 1D 算，用 einsum/broadcast 抬回 2D。
- 方向射线（task64）：**可分离**。水平填充 = (哪些行在 box 行带) ⊗ (该行 box边→最远点的列区间)。行带是 1D，列区间的端点每行一个值(1D)。
- 对称/裁剪/平铺/逐点：可分离或更简单。
- 连通性/flood-fill：**不可分离**（见 B5，改用 bit-packing）。

## B2. 【抬升】1D → 2D 不物化中间量
算完 1D 因子后，把它们组合成 2D fill：
- **广播比较**：`LessOrEqual(col_ramp[1,1,1,W], rmd[1,1,H,1]) → [1,1,H,W]`。一个算子出一个 2D 张量。
- **Einsum 外积**：`Einsum('bh,bw->bhw', rowfactor, colfactor)` 把行因子×列因子成 2D，一步、不物化中间。
- **原则**：2D 张量只在**最后一步**出现，且尽量只出 1 个（最终 fill），不要每个方向都物化 2D。

## B3. 【降维前先切片】Slice-to-actual-size
- 网格实际最大 24×24（task64 生成器 `randint(8,24)`），但张量是 30×30。
- 现网络 `Slice` 到 24×24 → 576 格，**每个全网格张量省 36%**，最后 `Pad` 回 30×30（Pad 输出可以是"output"或后续，边缘补 0 = padding 语义天然正确）。
- **每题先查生成器的 size 上界**，Slice 到该上界。免费的 36%+。

## B4. 【dtype】实测的硬约束（省几倍的关键）
- `input`/`output` 是 float32（grader 比较 `out>0`）。中间量尽量压。
- **CumSum 只支持 int32/int64/float**（实测：uint8/int8/int16 全报 INVALID_GRAPH）→ 8 个 CumSum 锁死 28800 地板，**这就是我 CumSum 版必败的原因**。避免 CumSum，用 ReduceMax/Min 极值法或 einsum。
- **ReduceMax/ReduceMin 支持 uint8**（实测 OK）→ 极值、投影用 uint8。
- 布尔逻辑用 `bool`(1字节)：Greater/Less/And/Or 直接出 bool。
- **别过早 Cast 到 float**：比较链全程 bool，只在最后必要处转。
- **本机 ORT 1.24.4 两个坑**：① group Conv 的 bias/filter 按组平铺；② 同进程第一个 Session 污染后续同构模型 → 多模型评测必须一模型一子进程（`validate.validate_isolated`）。

## B5. 【不可分离类】Bit-packing / SWAR
连通性、flood-fill、传播类**不可分离**，源头思想 = **SWAR（一个整数字里塞多个比特，字内并行）**：
- 一行 30 格布尔 → 一个 int（30 比特）。移位=空间平移，移位后 OR=传播/膨胀。900→~120 字节。
- 前缀/后缀 OR（"某侧是否有点"）= 倍增位移：`x|=x<<1; x|=x<<2; ...<<16`（5 步覆盖 30 位）。
- **难点（实测踩过）**：① float 打包只精确到 2^24，30 位要 2^29 → 溢出，须切两段或用 int64 conv/matmul 打包；② 移位跨行/跨段泄漏须 mask；③ 收敛步数要覆盖最坏尺寸。
- 已用于 task002/286/243（现网络里就是它）。

## B6. 【输出构造】别造 [1,10,H,W] 大张量
- 我的 56886 里三个 10 通道张量（in8+addend+out8 = 27000）是大头。
- 现网络用 `Einsum('bchw,ec->behw', ..., twohot)` + `Where`/`Scatter` 直接构造，避免逐通道 addend。
- 原则：输出的 10 通道结构，用 einsum 从 1D/紧凑因子**一步生成**到"output"（免费），不要 `input + fill*(e_dot−e_bg)` 这种全通道相加。

## B7. 决策树：拿到一题怎么打
```
1. 读生成器 → 确切规则 + size 上界 + 全输入空间
2. 分类计算：
   - 逐点/局部 → 单 Conv（多半已便宜，跳过）
   - 可分离（方向射线/对称/裁剪/带状） → B1 einsum投影 + B2 抬升 + B3 切片
   - 不可分离（连通/填充/传播） → B5 bit-packing
   - 全局聚合（计数/多数） → Reduce
3. dtype 全程压到 bool/uint8（守 B4 约束）
4. 输出用 einsum 一步构造（B6）
5. 验证：官方全样例 + `src/validate.py` 3000 fresh（防过拟合，task192 血教训）
6. 测成本 `tools/lean_score.py`，比现网络便宜且正确才收进 candidates/
```

## B8. 逆向拆解现网络的工作流（提取技术库的正道）
```python
# 对任一现网络：读 op 直方图 + 逐张量成本 + einsum 方程
m = onnx.load(path); g = onnx.shape_inference.infer_shapes(m).graph
# 1) Counter(nd.op_type)  -> 用了什么重算子
# 2) 逐 value_info 算 元素数×dtype字节 排序 -> 钱花哪
# 3) 打印每个 Einsum 的 equation 属性 -> 它怎么缩并/投影
# 4) 顺 producer 链看 1D 因子怎么组合成 2D
```
**这是唯一可靠的技术来源**（不是生成器聚类）。每读懂一个现网络的技术，加进技术库，就能迁移到它没覆盖的题。

## B9. task64 的具体下一步（把 56886 打到 ≤10358）
1. `Einsum('bchw,ec->beh'/'bew', input, dot_onehot)` 直接拿 dot 的行/列剖面（替掉我的 Gather+mask+极值）
2. box 的行/列剖面同理，从中取 box 行带/列带/边界（1D）
3. 射线区间：每行 `[box右边, 该行最远dot]` 从 1D 剖面算（rmd 已是 [1,1,H,1]，对了一半）
4. 只在最后用 1 个广播比较 + einsum 外积出 2D fill
5. Slice 到 24×24 全程；输出用 einsum 构造
6. 目标 ≤10358，floor-safe（兜底已锁）

---

# B10. 张量个数的坎：v1→v4 实证 + 平台硬边界（2026-07-14）

按 B9 真去啃了 task64（`tools/build_task064*.py`），全程 **267/0 正确**。实测曲线：

| 版本 | 用的技术 | cost | 主导成本 |
|---|---|---|---|
| v1 | float 朴素 + Gather 取通道 | 259642 → 162442 | 大张量 `[1,10,30,30]`（`input*e` 类 36000/个） |
| v2 | 极值法(ReduceMax)替代思路 + int8 输出 | 56886 | dtype 降下来，`addend[1,10,30,30]` 仍 36000→9000 |
| v3 | **Einsum 投影选色** + **OneHot 免费输出** | 51613 | 单张量小了，但**张量个数 82 个**暴露为瓶颈 |
| v4 | 删冗余 bg 门控（fewer tensors） | **45259** | 5 个 float einsum/reduce 输出(18000)+Mi int64(7200)+~60 小张量 |
| 现网络 | einsum 深度融合 | **10358** | ~30 张量、68 节点 |

**核心 insight：瓶颈随优化在迁移**——
- v1 阶段瓶颈 = **大张量**（[1,10,30,30] float）。杀手锏：Gather/Einsum 选通道、别造全通道积。
- v2/v3 阶段瓶颈 = **dtype**（float→int8/uint8/bool）+ **输出构造**（OneHot 单通道索引→免费 10 通道）。
- v4 阶段瓶颈 = **张量个数**。我物化 82 个中间张量，现网络只 ~30。**同样的逻辑，我一步步显式拆成独立张量，现网络用更少的重算子把逻辑折叠进去。**

**结论（可迁移的研究 insight）**：NeuroGolf 成本模型下，张量代数设计的目标**不是"每步逻辑清晰分解为独立张量"，而是"在保正确前提下把逻辑压进尽可能少的重算子"**。这本质是**图级编译器思维**——和"如何让高层逻辑压缩到最少的算子/状态周转"是同一个问题，直接迁移到 zero-BP / LLM graph-schedule / compiler-agent 类工作。

**平台硬边界（实测，意志力突破不了，设计时必须绕）**：
- `CumSum`：只支持 int32/int64/float（**uint8/int8/int16 报 INVALID_GRAPH**）→ 8 个 CumSum 锁死 28800，别用；用 ReduceMax/Min 极值法。
- `Einsum`：支持 int32/float，**不支持 uint8** → einsum 掩码输出至少 int32/float(3600)，压不到 uint8(900)。这是 v4 的 float 中间量下不去的原因。
- `ReduceMax/ReduceMin`：支持 uint8 ✅。
- `OneHot`：索引须 int64（int32 NOT_IMPLEMENTED），值须 float/int64（int8 不支持）；**索引 ≥ depth → 全零**（padding 用 99 哨兵）；带 slice 时 OneHot 输出是中间量(贵)，**不 slice 让 OneHot 直接当图输出=免费**。

**验收（诚实）**：路径"读电路→学技术→降成本"**实证有效（259642→45259 = 5.7×）**；但"**超过**专家深度融合的现网络 10358"**未达成**（仍 4.4× 高）。差距 = 图级算子融合深度，一个独立子技能。task64 candidates 里的 v1-v4 都比现网络贵 → 合并不选、**7268 地板不动**，兜底 `protected/task064_correct_v1_cost162442.onnx` 只读锁定。

---

# B11. 真技术库：现网络 task64 全 68 节点逆向（2026-07-14）

**关键认知**：所谓"公开 baseline"不是 naive 占位符，是**专家手工的紧电路**（全图仅 5 个 24×24 张量，cost 10358）。它用的算法比我的 v1-v4 更聪明——差在**算法层**，不是 cost-golf 技巧。以下 5 个技术是从它的电路里逆出来的，是真正可复用的武器库：

### T1. box vs dot 判别 = `count == area`
利用"实心矩形"的结构不变量：矩形的像素数 = 行跨度 × 列跨度；散点填不满包围盒。
```
rowproj = Einsum('bchw,ec->beh', input, cand_oh)   # 每候选色每行是否出现
rspan = ReduceSum(rowproj>0); cspan = ReduceSum(colproj>0)
area = rspan * cspan;  isbox = (cand_count == area)   # 零成本判别器
```
迁移：任何"实心形状 vs 稀疏标记"的判别都可用结构不变量（面积/周长/对称性），别用颜色频率长链。

### T2. 极值边界用 ArgMax 直接取（不用 mask×ramp+ReduceMax）
`M = Einsum('bchw,ec->behw', input, twohot)` 得 0/1 非背景网格（一次）。
```
L_i = ArgMax(M, axis=3)                     # 每行第一个非背景列（最左）
R_i = ArgMax(M, axis=3, select_last_index=1) # 每行最后一个（最右）
T_i/B_i = ArgMax(M, axis=2, ...)             # 每列上下边界
```
**一个 ArgMax 出一个边界向量**，省掉我 v2/v4 的 `mask×ramp`(900) + `ReduceMax` 两步/方向。

### T3. 单区间填充（每行一个 `[lo, lo+wid]`，不拆 4 方向布尔）
从 L/R 边界和 box 位置推每行填充区间，然后**一个 Sub + 一个 LessOrEqual 出整条方向 fill**：
```
lo_h = Max(L, notBoxRow*BIG)      # box行取最左非bg，非box行推出范围
wid_h = Min(R - lo_h, boxRow*BIG) # 区间宽度
Dh = col_coord - lo_h;  Hm = (Dh <= wid_h)   # 水平fill整片，1个2D张量
fill = Or(Hm, Vm)                  # 水平+垂直，共2个2D张量
```
对照：我 v4 拆成 rM/lM/dM/uM/hM/vM ~14 个布尔 2D 张量。这是"张量个数"差距的主因。

### T4. 坐标 +100 无分支排除 box
给 box 所在行/列的坐标加大常数（100），把它们顶出填充范围，替代显式 band 门控：
```
colb = col_coord + boxCol*100     # box列坐标变超大 -> 永不满足 <=wid
```
用算术代替逻辑分枝 = 少 mask 张量。

### T5. 输出 = 单个 `Where(fill, marker_onehot[1,10,1,1], input)`
```
output = Where(fill30[1,1,30,30], mk_sel[1,10,1,1], input[1,10,30,30])
```
广播选择器，**输出即图输出（免费）**。省掉我 v3 的 OneHot 索引网格(int64 7200) 或 v2 的 addend(9000)。

### 元结论（改写作战计划的地基）
- 公开 baseline = 专家紧电路。**模板化我自己的 v4 算法 = 造出更差的网络（负价值）；复现现网络 = +0；超过它 = 极难。**
- 差距在**算法层**：现网络"先选更聪明的算法(T1-T5)，再 cost-golf"；我"在自己的笨算法上 cost-golf"。
- **8000 的唯一 cost-golf 路径**：找到"当前最便宜网络 >> 结构可达下界"的题（gap≥3-5×），那里才有真空间。多数已被专家电路覆盖的题 gap 很小。→ 下一步 = **可达下界扫描**（用 T1-T5 估每题下界，对比现 best，筛 gap 大的）。模板的角色变成"放大已发明/逆向出的高级算法"，而非"复制现有算法"。

---

## ROI 提醒（给决策用）
单题攻下 ~+1 分（对数曲线），到 top5% 需 +37 ≈ **~40 题**，每题几小时 bit-level/einsum 工程 + 全尺寸验证。7268=top10% 已 Kaggle 锁定为地板。**先证明"读电路→学技术→超越 1 题"能走通，再决定刷不刷 40 题。**
