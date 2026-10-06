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

## 目录

```
toy_gpt/       自研被测模型（0.21M, val 1.76@108s）+ 3 组架构对照实验
               （无残差崩坏 / LN 三方案×深度 / 等参数深瘦vs浅胖：浅胖训练快 15×）
engine/
  no_cache_baseline.py   无 cache 贪心 + 计时
  v1_kv_cache.py         朴素 KV Cache（玩具 cat 版/预分配版 + Qwen HF 版 + bench）
  v2_two_stage.py        两阶段引擎 + 可插拔采样（pick 插槽）+ EOS 自然停（chat 模式）
benchmarks/    测量协议（热身遍/同步掐表/单变量清场——冷启动税 17× 的教训）
```

## 环境与运行

- 依赖：`torch`（CUDA）· `transformers>=5`（v5 的 `DynamicCache` 新 API：`cache.layers[i].keys/.values`）
- 模型自动定位：优先本地 `D:/实习/models/Qwen3-0.6B`，不存在则回退 HuggingFace id `Qwen/Qwen3-0.6B`（自动下载）
- 一键自测（三连全绿 = 两阶段+采样语义无损）：

```bash
python engine/v2_two_stage.py
# ① greedy == 无 cache 基线  True   ② 同 seed 逐字复现  True   ③ 换 seed 不同  True
```

- 生成演示（自定义 prompt 与采样参数）：

```bash
python demo.py "从前有一座山，山里有座庙，" --n 50 --t 0.8 --p 0.9 --seed 42   # 裸续写
python demo.py "用一句话解释什么是 KV Cache" --chat                          # 对话模式：chat template + EOS 自然停
```

## Roadmap

- **v3**：KV Cache int8/int4 量化 + 率失真曲线（比特-显存-生成质量 tradeoff）
- continuous batching 简化版（W4 学 nano-vllm 后回填）
