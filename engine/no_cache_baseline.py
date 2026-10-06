# -*- coding: utf-8 -*-
"""no_cache_baseline.py —— 无 KV Cache 的贪心基线 + 计时（Qwen3-0.6B）
实测（5070 Ti, bf16, prompt 2000, 热身后）：~206 ms/token（对照 v1 cache 34 ms ≈ 6x）"""
import os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_LOCAL = 'D:/实习/models/Qwen3-0.6B'
M = _LOCAL if os.path.isdir(_LOCAL) else 'Qwen/Qwen3-0.6B'   # 本地优先，否则走 HF（自动下载）

@torch.no_grad()
def greedy_nocache(model, ids, n_new):
    ids = ids.clone()
    for _ in range(n_new):
        logits = model(ids).logits[:, -1]
        nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
    return ids

if __name__ == '__main__':
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to('cuda').eval()
    ids = tok('The capital of China is', return_tensors='pt').input_ids.cuda()
    out = greedy_nocache(model, ids, 10)
    print(tok.decode(out[0]))
    greedy_nocache(model, ids, 3)                      # 热身
    torch.cuda.synchronize(); t0 = time.perf_counter()
    greedy_nocache(model, ids, 20); torch.cuda.synchronize()
    print(f'TPOT(无cache): {(time.perf_counter()-t0)/20*1000:.1f} ms/token')
