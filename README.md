# NeuroGolf 2026 — ARC→最小 ONNX 电路编译器

把 ARC-AGI-1 的 400 个训练任务，各编译成一个满足赛题约束的**最小静态 ONNX 网络**。
评分 `per_task = max(1, 25 − ln(memory + params))`，总分 400 题之和，满分 10000。

> 完整战略与逐日计划见 [PLAN.md](PLAN.md)。本文档是工具链与方法学的工程说明 + 已验证结论。

## 目录结构

```
data/                     官方数据：task001-400.json + neurogolf_utils.py(官方评分器)
third_party/arc-gen/      官方生成器：每题一个 Python 文件 = 真值规则 + 私有分布采样器
third_party/arc-dsl/      Hodel 的 400 题符号 solver（280 题在全样例通过）
baselines/udit/           最强公开 bundle(LB 7243)：nets/ + scores_isolated.json(可信逐题分)
src/
  oracle.py               官方评分复刻：evaluate(model, task) -> {points, cost, memory, params}
  graphlib.py             onnx.helper 建图工具 + static_cost() + 常用单节点电路工厂
  validate.py             三级验证：官方样例 + N个新生成样例 + 约束审计；validate_isolated()
  compile/                逐题手工电路（fix4.py 等）
tools/
  gen_examples.py         调 arc-gen 生成器批量产新样例（私有集同分布），缓存 gen_cache/
  batch_score.py          每模型一个子进程的批量评分（绕开 ORT 会话污染）
  refit_conv.py           单 Conv 整数权重 LP 拟合 + 核裁剪（局部线性可分题）
  fit2layer.py            两层整数电路拟合（局部非线性可分题，torch）
  refit_all.py            批量 refit + 验证 + 只收真赢 baseline 的
  package.py              候选池择优 -> submission.zip
experiments/              成本实验、优先级表、局部性清单、攻坚清单
candidates/taskNNN/       每题候选池：*.onnx + *.json(验证报告，ok=true 才入池)
submissions/              打包产物
```

## 环境（必须与官方评分栈对齐）

```
~/venvs/neurogolf/bin/python   # onnx 1.21.0, onnxruntime 1.24.4, numpy 2.4.4, onnx-tool 1.0.1, scipy, torch
```
项目盘是 exFAT，venv 不能放盘内（放 ~/venvs/）。

## 标准工作流

```bash
# 1. 读规则（真值就是源码）
cat third_party/arc-gen/tasks/task_<arcid>.py

# 2. 建电路（手工用 graphlib，或自动用 refit_conv/fit2layer）
python tools/refit_conv.py <N> 7          # 尝试单 Conv
python tools/fit2layer.py  <N> <r>        # 尝试两层

# 3. 三级验证（官方样例 + 3000 新样例 + 约束）
python src/validate.py <model.onnx> <N> --fresh 3000

# 4. 只有 ok=true 且 points > baseline 才把 .json 报告留在 candidates/taskNNN/
# 5. 打包提交
python tools/package.py <tag>             # 生成 submissions/submission_<tag>.zip
cp submissions/submission_<tag>.zip submissions/submission.zip   # 文件名必须叫 submission.zip
kaggle competitions submit neurogolf-2026 -f submissions/submission.zip -m "..."
```

## 成本模型的物理规律（实测验证）

- `cost = params + memory`。`params` = initializer/Constant 元素个数（与 dtype 无关；共享只计一次）。
  `memory` = 所有**中间张量**字节数（静态 shape × dtype 宽，profiler 取峰值）。
- **`input`/`output` 张量免费；节点属性免费。** → 单节点电路 = 0 内存 = 只算参数。
- 中间张量 dtype：bool 1B / int8 1B / fp16 2B / fp32 4B / int64 8B。一个 fp32 全网格 [1,10,30,30]=36KB→14.5 分。
- **早收缩晚展开**：大张量只在首末节点；内部 reduce 到 [1,10,1,1]=40B 后逻辑近乎免费。
- 老 opset 属性版算子（Slice-1/Pad-2/Upsample-7/带 axes 的 Reduce*）零参数；每题可自选 opset。
- `Where` 合法，`If` 因子图属性被禁。sparse_initializer 触发官方 sanitize bug，禁用。
- 输出只需**符号正确**（`(out>0)` 判分，官方 `run_network` 原文）→ 可用 hinge 松弛拟合后量化。

## 两个本机 ORT 1.24.4 坑（血泪）

1. **group Conv 的 bias 和 filter 行按组平铺**（第二组错误复用第一组）——设计权重须在规范/平铺两种语义下都符号正确。
2. **同进程内第一个 InferenceSession 污染后续同构模型**（算成第一个的函数）——一切批量评测走子进程（batch_score.py / validate_isolated）。曾误判 baseline 有 4 题失败，实际只有 1 题。

## 已验证的核心结论（别再重复踩）

1. **baseline 7243 是社区 3 个月精调的强基线**，平均 18.1 分/题，贴近很多任务的物理 cost 下界。
2. **评分漏洞路已封死**：早期 >9000 分是漏洞驱动、已被主办方重评打回；现在榜首 ~8000 是合法 golfing。逆向审计评分器（sanitize 名字冲突检查、profiler 反作弊、封 Compress/sparse/子图）无残余可利用漏洞。
3. **前 10（7920）合法可达但需几百题深度 golfing**（社区 3 个月的量级），剩余时间内不现实。现实合法天花板 ~7300–7500。
4. **自动化低产**：几何/对称核裁剪赢不了 baseline（它做过非对称 pad 精调）；11 个严格局部题里单 Conv 只赢 1 个（task192，其余非线性可分）；普通两层 Conv 因 float 中间张量太贵反而更差（需 int8/QLinearConv）。
5. **真正局部性必须在 ≥3000 新样例上验**（官方 ~265 样例会假阳性，见 truly_local vs strict_local 的半径差异）。
6. **抗过拟合是生命线**：拟合样例 <3000 会过拟合（task192 首拟合 609/2000 新样例崩，被 gate2 拦截）。

## 未尽的高价值方向（交接）

- **int8/QLinearConv 两层电路**：10 个真局部非线性可分题（243/077/004/278/265/359/208/162/070/222）的正解，中间张量全程 int8（H×900），潜在 +2~+10。fit2layer.py 已有 torch 拟合框架，需把 build() 从"Cast包裹的float Conv"改成真正的 QLinearConv 链。
- **最差40手工攻坚**：headroom 最大（到18共 ~99 分）但多需真实空间计算（enclosure/ray-cast/对象/计数），成功率低、单题耗时。见 experiments/attack_list.json。
- **arc-dsl primitive→模板 translator**：280 题有符号 solver，≤5 行的 ~80 题可半自动编译。
