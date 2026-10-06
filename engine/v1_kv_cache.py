"""
v1_kv_cache.py —— 朴素 KV Cache 引擎（W2 产物，2026-10）
================================================================================
为什么加速比是这个数（实测 5070 Ti，Qwen3-0.6B bf16，prompt 2000）：
  无 cache 203.8 ms/token vs 手写 cache 33.4 ms/token = 6.1×
  玩具侧（0.22M GPT）实测 0.69×（cat 版）/ 0.85×（预分配版）——launch-bound：
  前向耗时随 T 几乎不变（E1），省计算救不了派活钱；cat 税 1.77ms/步与两版
  差额对账吻合（E2）。真药方 = batch 摊薄（W4）/ CUDA Graph（W5）。
================================================================================
"""
import time, torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


# ---------- 玩具版（0.22M GPT，model 传入，账本闭包化——接口卫生版） ----------
@torch.no_grad()
def gen_cache_toy(m, ctx, n_new):
    """朴素 cat 版：每层每头 (ks, vs) 列表；先 append 后 cat（一读一写）。"""
    n_l, n_h = len(m.blocks), len(m.blocks[0].sa.heads)
    cache = [[([], []) for _ in range(n_h)] for _ in range(n_l)]   # 每场新建

    def step(tid, pos):                                   # pos = 绝对位置（头号坑）
        x = m.tok(tid) + m.pos(torch.tensor([[pos]], device=tid.device))
        for l, blk in enumerate(m.blocks):
            h = blk.ln1(x); outs = []
            for hd, head in enumerate(blk.sa.heads):
                k, q, v = head.key(h), head.query(h), head.value(h)
                ks, vs = cache[l][hd]
                ks.append(k); vs.append(v)                # 一写
                K = torch.cat(ks, 1); V = torch.cat(vs, 1)   # 一读（O(t)，D3 已审）
                wei = F.softmax(q @ K.transpose(-2, -1) * head.hs ** -0.5, -1)
                outs.append(wei @ V)                      # 免掩码：唯一 query=最新位置
            x = x + blk.sa.proj(torch.cat(outs, -1))
            x = x + blk.ffwd(blk.ln2(x))
        return m.head(m.lnf(x))[:, -1]

    ids = ctx.clone()
    for t in range(ids.shape[1]):                         # 朴素 prefill：逐字喂
        logits = step(ids[0, t].view(1, 1), t)
    for i in range(n_new):                                # decode：贪心
        nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        if i < n_new - 1:                                 # guard：最后一张没人读
            logits = step(nxt, ids.shape[1] - 1)
    return ids


# ---------- Qwen 版（HF past_key_values，prefill 整段 + decode 单 token） ----------
@torch.no_grad()
def greedy_hf_cache(model, ids, n_new):
    """DynamicCache 账本；prefill 整段一次（并行），decode 只前向新 token。"""
    ids = ids.clone()
    cache = DynamicCache()                                # 每场新建：上一场的旧账会串味
    logits = model(ids, past_key_values=cache, use_cache=True).logits[:, -1]   # prefill：整段一次=全位置并行
    for i in range(n_new):
        nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        if i < n_new - 1:                                 # guard：最后一张打分表没人读，省一次前向
            logits = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1]   # decode：只喂 (1,1)
    return ids


# ---------- 计时（Day 7 三纪律：热身 / synchronize / 多遍均值） ----------
@torch.no_grad()
def bench_cache(model, ids, n=20):
    cache = DynamicCache()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model(ids, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize(); ttft = (time.perf_counter() - t0) * 1000
    logits = out.logits[:, -1]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for i in range(n):
        nxt = logits.argmax(-1, keepdim=True)
        logits = model(nxt, past_key_values=cache, use_cache=True).logits[:, -1]
    torch.cuda.synchronize()
    return ttft, (time.perf_counter() - t0) / n * 1000    # (TTFT ms, TPOT ms)
