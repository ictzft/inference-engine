# inference-engine —— LLM 推理引擎从零实现与性能优化

从零手写轻量推理引擎（KV Cache → 两阶段调度 → 采样全家桶），**每一步优化都用实测数字量化**。双平台验证：自训 0.21M 字符级 GPT + Qwen3-0.6B（bf16）。

## 核心数字（5070 Ti 实测，测量协议见 [benchmarks/timing_protocols.md](benchmarks/timing_protocols.md)）

| 优化 | 实测 | 说明 |
|---|---|---|
| **KV Cache（Qwen3-0.6B, prompt 2000）** | 无 cache 206.1 → 34.2 ms/token = **6.0×**（同机复测 7.2×） | 与官方 generate 逐 token 全等；手写 HF `past_key_values` 版 |
| KV Cache（玩具 0.21M GPT） | **0.69×（cat 版）/ 0.85×（预分配版）——更慢** | launch-bound 实证：前向耗时随 T 几乎不变（省计算救不了派活钱）；cat 税 1.765ms/步与两版差额对账吻合 |
| 两阶段拆分（v2） | TTFT 208.1 / TPOT 28.6 ms，**加速比 7.2×** | prefill compute-bound（时间随 T 线性+T² 项）vs decode launch 主导（TPOT 随 T 全平）——拆分零语义漂移（与官方 generate 逐位全等） |
| 采样全家桶 | temperature/top-k/top-p/rep-penalty 与 transformers warper **逐位 allclose** | 极限恒等式验证：T→0 ≡ argmax、k=1 ≡ argmax、p→0 ≡ 贪心；同 seed 逐字复现 / 换 seed 必变（Generator 专用骰子） |

## 全套手算对账（Qwen3-0.6B，误差 0%）

参数 596,049,920（emb 26% + 28 层 × 15.7M ≈ **15·d²**）→ bf16 权重 **1.192 GB**（=torch allocated 分毫不差）→ 每 token KV **4KB/层 · 112KB/模型**（=实测 114,688B 分毫不差；GQA 16Q/8KV 账单五折）→ @40960 顶格 4.70GB（权重 3.9 倍）→ 屋顶线 1.46ms（896GB/s）vs 实测 30ms ≈ **20×**（launch 主导，0.6B 未到 memory-bound）。

## 目录（每个文件的作用）

```
inference-engine/
├── demo.py                        # 一行命令生成演示：prompt/--n/--t/--p/--seed/--chat（裸续写 vs 对话模板+EOS 自然停）
│
├── engine/                        # ★ 主角：三版引擎演进（no_cache 量化浪费 → v1 消除浪费 → v2 完整引擎）
│   ├── no_cache_baseline.py       #   v0 基线：无 cache 贪心循环（每圈整段重喂=重做 prefill 的活）
│   │                              #      ——所有加速比的分母；O(N²) 浪费的量化器（prompt 10→4000：34.8→801.8 ms/token）
│   ├── v1_kv_cache.py             #   v1：KV Cache——玩具手写 cat 版（账本=每层每头 (ks,vs) 列表）
│   │                              #      + Qwen HF past_key_values 版 + TTFT/TPOT 计时器；Qwen3 实测 6.0×，玩具 0.69×（launch-bound）
│   └── v2_two_stage.py            #   v2：两阶段引擎——prefill/decode 显式拆分（TTFT/TPOT 分开测）
│                                  #      + pick 可插拔采样（与 vLLM 语义逐位对账）+ EOS 自然停 + 出厂自测三连
│
├── toy_gpt/                       # 被测模型：0.21M 字符级 GPT（实验台；93s 可重训，val 1.76）
│   ├── model.py                   #   Head/MHA/FFN/Block/GPT 逐件实现；Block 可切 ln_mode/norm_pos（为对照实验留缝）
│   ├── train.py                   #   训练入口：5000 步 ~92s → val 1.76；seed 1337 固定=任何人重跑分毫不差
│   ├── input.txt                  #   字符级语料（~1MB，65 字符词表）
│   └── experiments/               #   3 组架构对照实验（每份自带用法 docstring）
│       ├── exp1_no_residual.py    #     去掉残差：loss 卡死 3.35、底层梯度≈顶层 1/3.5——残差流的价值
│       ├── exp2_ln_variants.py    #     一个 Block 为什么要两个 LayerNorm：标准/共享/单入口三方案 + 激活统计探针
│       └── exp3_arch_ablation.py  #     A：LN 方案×深度（4/8/12 层）｜B：post-norm 三连（付 0.02 梯度税未崩）
│                                  #     ｜C：等参数深瘦 vs 浅胖（浅胖训练快 15×：kernel 启动开销实证）
│
└── benchmarks/
    └── timing_protocols.md        # 测量协议：热身遍 / synchronize 前后夹住 / 单变量清场（冷启动税 17× 的教训）
                                   #      ——本 README 全部数字的可信度来源
```

**推荐阅读顺序**（面试官视角）：`demo.py` → `engine/v2_two_stage.py`（主角）→ `v1_kv_cache.py`（v2 的上一代）→ `no_cache_baseline.py`（分母）→ `toy_gpt/model.py`（手写版 attention 与真模型同构对照）。

## 环境与本地测试手册

**环境（作者本机）**：Windows + RTX 5070 Ti · Python 3.11 conda 环境（torch cu128、transformers 5.17）。本机运行用 `E:\anaconda\envs\infra\python.exe`（下文简写 `python`）；任意机器装好 `torch`(CUDA) + `transformers>=5` 即可。模型自动定位：本地 `D:/实习/models/Qwen3-0.6B` 优先，不存在则回退 HuggingFace id `Qwen/Qwen3-0.6B`（自动下载约 1.2GB）。

### 测试 1｜引擎自测（两阶段+采样语义无损三连）

```bash
python engine/v2_two_stage.py
```

预期输出（2026-10-06 实测）：

```
① greedy == 无cache 基线: True
② 同 seed 逐字复现: True | ③ 换 seed 不同: True
样例: ' Beijing, the capital of Russia is Moscow, and the capital'
```

### 测试 2｜生成演示（demo.py）

完整模板（全部参数可省，光 `python demo.py` 也能跑）：

```
python demo.py [prompt] [--n N] [--t T] [--p P] [--seed SEED] [--chat]
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `prompt` | `'The capital of China is'` | 提示词（位置参数，直接写在命令后） |
| `--n` | 50 | 生成 token 上限（防爆阀；`--chat` 下由 EOS 决定实际停点） |
| `--t` | 0.8 | temperature：低温固定，高温发散 |
| `--p` | 0.9 | top-p 截断 |
| `--seed` | 不给 | 给定则逐字可复现（同命令两遍输出全同） |
| `--chat` | 关 | 对话模式：chat template + EOS 自然停（开关型，**后面不跟值**） |

常用配方：

```bash
python demo.py "中国的首都是哪里？" --chat                                # 对话 + EOS 自然停
python demo.py "用一句话解释什么是 KV Cache" --chat --t 0.6 --seed 42    # 稳定且可复现
python demo.py "Once upon a time" --t 0.3 --n 80                         # 低温：稳但同质
python demo.py "Once upon a time" --t 1.5 --n 80                         # 高温：野但胡话风险
python demo.py "从前有一座山，山里有座庙，" --n 60                        # 裸续写
```

预期要点：`--chat` 回答干净无模板残渣（如"中国的首都是北京。"5 token 自然停，不跑满 n）；每发尾部带 `[n tokens / s = tok/s（含 TTFT）]` 统计；`--seed` 同命令跑两遍输出逐字相同。

### 测试 3｜无 cache 基线计时（对照组）

```bash
python engine/no_cache_baseline.py
```

预期：贪心输出 + `TPOT(无cache): ~34 ms/token`（短 prompt；对照 v2 两阶段 TPOT 28.6ms——prompt 越长差距越大，核心数字表的 6.0×/7.2× 即此差）。

### 测试 4｜被测模型复训（0.21M 字符级 GPT）

```bash
python toy_gpt/train.py 200     # 快速自检 ~10s
python toy_gpt/train.py         # 全量 5000 步
```

预期（2026-10-06 全量实测）：`参数量: 209729`（=手推对账 12·d² 主项）→ step 5000 `val 1.7617`，耗时 92~108s @5070 Ti。

### 常见坑（Windows）

- CMD 跨盘切目录必须 `cd /d D:\...`（不带 `/d` 不生效）；
- 系统 Python 缺依赖 → 用 conda 环境完整路径；
- `--n` 是防爆上限不是目标长度：`--chat` 由 EOS 决定停，裸续写跑满 n；
- v5.17：`apply_chat_template(tokenize=True)` 返回 BatchEncoding（demo.py 已按实测处理）。

## Roadmap

- **v3**：KV Cache int8/int4 量化 + 率失真曲线（比特-显存-生成质量 tradeoff）
- continuous batching 简化版（W4 学 nano-vllm 后回填）
