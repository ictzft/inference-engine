# -*- coding: utf-8 -*-
"""no_cache_baseline.py —— 无 KV Cache 的贪心基线 + 计时（Qwen3-0.6B）
实测（5070 Ti, bf16, prompt 2000, 热身后）：~206 ms/token（对照 v1 cache 34 ms ≈ 6x）
浪费机理：每圈把【整段序列】重喂 → 历史 K/V 全部重算（= 重复做 prefill 的活，O(N²)）。
浪费随长度滚雪球：prompt 10→4000 token，单 token 延迟 34.8→801.8 ms（平→线性→平方三段曲线）。"""
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_LOCAL = 'D:/实习/models/Qwen3-0.6B'
M = _LOCAL if os.path.isdir(_LOCAL) else 'Qwen/Qwen3-0.6B'   # 本地优先，否则走 HF（自动下载）

@torch.no_grad()
def greedy_nocache(model, ids, n_new):
    """最笨但最无疑的生成循环——整个项目所有加速比的分母（对照组）。"""
    ids = ids.clone()                       # 防御性起点：不碰调用者的张量
    for _ in range(n_new):
        logits = model(ids).logits[:, -1]   # 每圈整段重喂：前 T-1 张打分表全白算，只取最后一张
        nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)      # 序列越喂越长 → 每圈越来越慢（O(N²) 的现场）
    return ids

if __name__ == '__main__':
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to('cuda').eval()
    ids = tok('The capital of China is', return_tensors='pt').input_ids.cuda()
    out = greedy_nocache(model, ids, 10)
    print(tok.decode(out[0]))
    greedy_nocache(model, ids, 3)                      # 热身遍：烧掉冷启动税（首遍可比稳态慢 17×），不计入
    torch.cuda.synchronize(); t0 = time.perf_counter() # synchronize 前后夹住：CUDA 异步，不同步量到的是递交时间
    greedy_nocache(model, ids, 20); torch.cuda.synchronize()
    print(f'TPOT(无cache): {(time.perf_counter()-t0)/20*1000:.1f} ms/token')
