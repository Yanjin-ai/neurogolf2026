# NeuroGolf 电路编译作战手册（编译 agent 必读）

你的任务：给定一个 taskNNN，把它的真实规则编译成**最小成本**的 ONNX 网络，
严格通过三级验证后落盘为候选。当前 baseline 分数见 `experiments/workqueue.json`，
你的产出必须**严格高于 baseline 分**才有价值（等分不如 baseline 稳，不收）。

## 0. 环境与路径（一律绝对路径）

- 仓库根：`/Volumes/SANDISK ELE/ARCAGIneurogolf`（下文 $ROOT）
- Python：`~/venvs/neurogolf/bin/python`（onnx 1.21.0 / onnxruntime 1.24.4，对齐官方）
- 任务数据：`$ROOT/data/taskNNN.json`（train/test/arc-gen 三组样例）
- **真实规则源码**：`$ROOT/third_party/arc-gen/tasks/task_<arc_id>.py`（arc_id 见 workqueue）
  —— 这是官方生成器，读 `generate()` 就是读规则本身；`common.py` 里有工具函数定义
- 参考程序：`$ROOT/third_party/arc-dsl/solvers.py` 里 `solve_<arc_id>`（workqueue 里 dsl_ok=True 才可信）
- 建图工具：`$ROOT/src/graphlib.py`（GraphBuilder / static_cost；也可直接用 onnx.helper）
- 产出位置：`$ROOT/candidates/taskNNN/<你起的名>.onnx` + 同名 `.json`（验证报告，必须 ok=true）

## 1. 评分物理定律（决定一切设计）

`score = max(1, 25 − ln(max(1, memory + params)))`，仅当全部样例+私有样例全对才得分。

- `params` = initializer/Constant 的**元素个数**（与 dtype 无关，标量=1，共享 initializer 只计一次）
- `memory` = **中间张量**字节数之和（静态 shape × dtype 宽度）。**名为 input/output 的张量免费，节点 attribute 免费**
- 推论：
  - 单节点图 = 只有 params 成本；N 节点链 = N−1 个中间张量计费
  - **早收缩晚展开**：大张量只允许出现在首/末节点；中间只留标量/小向量
  - dtype 费率：bool 1B，fp16 2B，fp32 4B，int32 4B，int64 8B（Gather 索引用 int32）
  - 一个 fp32 [1,10,30,30] 中间张量 = 36000 → 直接掉到 ≤14.5 分，等于宣判死刑
  - cost 参考：1→25 分，4→23.6，10→22.7，30→21.6，148→20，1097→18，8100→16

## 2. I/O 语义（必须烂熟）

- 输入 `input`：float32 [1,10,30,30]，one-hot（通道=颜色 0..9，**黑色 0 也是 one-hot**），
  网格置于左上角，网格外全零（zero-hot）
- 输出 `output`：同形状；判分为 `(out > 0.0)` 与目标 one-hot 逐元素相等
  —— **只需符号正确**：正确通道 >0（多大都行），其他通道与 padding 全部 ≤0（0 也算否）
- **一热恒等式**：任意在格单元 Σ_c input[c,r,w] = 1，padding = 0
  → "在格指示器/网格尺寸"都是输入的线性函数（ReduceSum/ReduceL2 提取）
- 输出网格尺寸由"尾部全零行列"隐式界定；输出比输入小/大时，多余区域必须全 ≤0

## 3. 原语手册（实测成本，优先从上往下套）

| 原语 | cost | 用途 |
|---|---|---|
| 单节点属性算子 Transpose/Slice-1/Pad-2/Upsample-7/Pool/DepthToSpace | 0 | 几何、裁剪、平铺、缩放 |
| 自门控 Einsum（如 `nchw,nkwh->nchw`） | 0 | 恒等/对称类 |
| **RoiAlign**（rois[1,4]+batch_idx[1]，负坐标可翻转） | 5 | 任意固定 crop/flip/scale 采样，attributes 里 output_h/w 免费 |
| Slice（starts/ends/axes/steps 各 1 元素） | 4–8 | 镜像、rot180（固定尺寸时） |
| Gather(axis=1, idx[10] int32) | 10 | 颜色重映射 |
| Gather(axis=2/3, idx[30] int32) | 30 | 固定行列置换/平移 |
| ReduceL2 全张量 → 标量 = √(h·w)（正方形网格边长） | ~8 | 尺寸提取进标量域 |
| Range(start,limit,delta)+Gather 负索引回卷 | ~138 | 变尺寸镜像/平移 |
| Reduce 至 [1,10,1,1] 后的标量逻辑（Greater/Where/Equal） | 40–150 | 计数、比色、全局条件 |
| Einsum 选择矩阵 (30,s)+(s,s) | ~s²+30s | 小固定网格行列变换 |
| 单 Conv 3x3 [10,10,3,3]（可加 bias[10]） | 900/910 | 任意局部规则（光环/边界/角点检测） |
| Conv group=2 [10,5,3,3] | 460 | 依赖不跨 0-4/5-9 通道组时 |
| MaxPool k3s3 / AveragePool + Pad-2 | ~4000 | 整块缩放、块统计 |
| ReduceMax(ch)+GreaterOrEqual 末节点 WTA | 3600 | 赢者通吃重编码 |
| bool/fp16 网格域形态学链 | 9000+/层 | 区域、包围、bbox |
| 展开迭代传播（共享权重 Conv+Min，步数按任务实测最大距离+2） | ≥2000/步 | 洪泛/连通域，最后手段 |

组合技巧：
- 条件逻辑 `Where` 合法（`If` 被禁）；条件先 Reduce 到标量再比较
- Conv 天然在边界截断（光环出格自动裁剪）；负 pads 属性可免费裁剪
- 变尺寸输出：用行/列占用掩码乘回去，或 Gather 负索引回卷到 padding 零区
- opset 可每题自选一个版本：Slice-1/Pad-2/Upsample-7 要老 opset（≤9）；
  Equal(float)≥11、GreaterOrEqual≥12、Einsum≥12——不能兼得时换算子或换 opset

## 4. 工程纪律（违反=白干）

1. **整数权重、双侧 margin ≥1**。禁止实数拟合贴 0 边缘（私有集杀手）
2. **padding 泄漏自查**：网格贴边时，卷积/位移会把信号漏进 padding 区。
   每个正权重通道都要推演"padding 单元格会不会被推成 >0"。用一热恒等式锚定
3. **ORT 1.24.4 双坑**：
   - group Conv 的 bias 和 filter 行按组平铺（out5-9 复用 w[0..4]/b[0..4]）。
     用 group 时权重必须在"规范语义"和"平铺语义"下都符号正确（参考 `src/compile/fix4.py` task352）
   - 同进程第一个 InferenceSession 会污染后续同构模型 → **验证只用下面的 CLI**（天然每次一进程），
     自己写调试脚本时一个进程只碰一个模型
4. 禁：Loop/Scan/NonZero/Unique/Compress/*Sequence*/If/子图/sparse_initializer/多输入输出/动态 shape
5. `onnx.helper` 直接建图，ir_version=10；不要用 PyTorch 导出
6. 文件放 `$ROOT/candidates/taskNNN/`；忽略 `._*` 垃圾文件（exFAT）

## 5. 验证协议（一步不可省）

```bash
cd "$ROOT" && ~/venvs/neurogolf/bin/python src/validate.py \
  "$ROOT/candidates/taskNNN/<name>.onnx" NNN --fresh 2000
```
- 输出 JSON：`ok=true` 且 `points` > baseline 才算成功
- 失败时报告里有 gate 信息；用 `data/taskNNN.json` 的样例自己写小脚本对拍差异单元格
  （一个进程只建一个 InferenceSession！）
- 成功后把验证 JSON 原样存为 `.onnx` 同名 `.json`（package.py 靠它择优）
- 拿不下也别硬撑：确认规则理解正确、尝试 ≥2 种架构后仍不能低于 baseline cost 的一半，
  就写一份 `candidates/taskNNN/NOTES.md` 记录你对规则的理解、尝试与失败原因，撤退

## 6. 设计流程建议

1. 读 arc-gen 生成器源码 → 用一句话写出规则；数清：输入输出尺寸关系？颜色集合？
   随机参数范围（尺寸/数量/位置边界，决定 padding 风险与步数上界）
2. （dsl_ok=True 时）读 arc-dsl solver 交叉印证结构
3. 从原语手册**自上而下**套：能单节点吗？能标量域吗？能单 Conv 吗？…
4. 写权重时逐 case 推演符号（在格/出格/贴边/重叠/极端随机参数）
5. 建图 → 验证 CLI → 看失败单元格 → 修 → 直到 ok=true
