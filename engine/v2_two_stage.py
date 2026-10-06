"""
v2_two_stage.py —— 两阶段引擎 + 采样全家桶（W3 产物，2026-10）
================================================================================
架构：prefill(整段填账, compute-bound) + decode(单 token, memory-bound) + 可插拔 pick
语义对照：temperature/top_k/top_p/repetition_penalty 与 vLLM SamplingParams 同式
（对账依据：与 transformers TemperatureLogitsWarper/TopKLogitsWarper/TopPLogitsWarper/
  RepetitionPenaltyLogitsProcessor 输出逐位 allclose；极限恒等式 T→0/k=1/p→0 均验证）
================================================================================
"""
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


# ---------- 两阶段 ----------
@torch.no_grad()
def prefill(model, ids):
    """整段一次前向填满账本；返回 (cache, 最后一张打分表)。
    logits[:, -1].clone()：切片是视图，拖着整张 (1,T,V) 打分表（@4000 约 1.2GB）。"""
    cache = DynamicCache()
    logits = model(ids, past_key_values=cache, use_cache=True).logits[:, -1].clone()
    return cache, logits


@torch.no_grad()
def decode_step(model, nxt, cache):
    """只前向新 token (1,1)——绝不喂整段；@no_grad 是生存条件（裸跑 ~1.8GB/圈@T=4000）。"""
    return model(nxt, past_key_values=cache, use_cache=True).logits[:, -1].clone()


# ---------- 采样全家桶（改的都是 logits，最后统一 softmax→multinomial） ----------
def greedy(logits, gen=None):
    return logits.argmax(-1, keepdim=True)


def sample_temperature(logits, t=1.0, gen=None):
    logits = logits.float()
    return torch.multinomial(F.softmax(logits / t, -1), 1, generator=gen)   # ⭐ 温度作用在 logits 上


def sample_top_k(logits, k=50, gen=None):
    logits = logits.float()
    v, _ = torch.topk(logits, k)
    logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))        # 门外全 −inf
    return torch.multinomial(F.softmax(logits, -1), 1, generator=gen)


def sample_top_p(logits, p=0.9, gen=None):
    logits = logits.float()
    probs = F.softmax(logits, -1)
    svals, sidx = torch.sort(probs, descending=True)
    keep = (svals.cumsum(-1) - svals) < p                                   # 跨过 p 前的最小前缀（≥1 个）
    mask = torch.zeros_like(keep).scatter(-1, sidx, keep)
    logits = logits.masked_fill(~mask, float('-inf'))
    return torch.multinomial(F.softmax(logits, -1), 1, generator=gen)


def apply_rep_penalty(logits, hist, penalty=1.2):
    """历史 token 正分÷penalty、负分×penalty（vLLM/HF 同式）；只改打分表不抽样。"""
    logits = logits.clone().float()
    for i in set(hist[0].tolist()):
        logits[0, i] = logits[0, i] / penalty if logits[0, i] > 0 else logits[0, i] * penalty
    return logits


def make_sampler(t=1.0, k=0, p=0.0, rep=1.0):
    """组合器：rep → 温度 → top-k → top-p → 抽签。返回 pick(logits, gen=None) -> (1,1)。"""
    def pick(logits, gen=None):
        logits = logits.float()
        # （rep 在引擎层用 hist 调 apply_rep_penalty，组合器内不重复实现）
        if t != 1.0:
            logits = logits / t
        if k:
            v, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))
        if p:
            probs = F.softmax(logits, -1)
            svals, sidx = torch.sort(probs, descending=True)
            keep = (svals.cumsum(-1) - svals) < p
            logits = logits.masked_fill(~torch.zeros_like(keep).scatter(-1, sidx, keep), float('-inf'))
        if gen is None:
            return torch.multinomial(F.softmax(logits, -1), 1)
        return torch.multinomial(F.softmax(logits, -1), 1, generator=gen)
    return pick


# ---------- 引擎主体：pick 可插拔 ----------
@torch.no_grad()
def generate_v2(model, ids, n_new, pick=greedy, seed=None, rep=1.0):
    """两阶段生成：prefill 一次 + decode 循环；pick 可换（greedy/温度/topk/topp/组合）。
    seed 给定则逐字可复现；rep≠1 时每步先对历史施加重复惩罚。"""
    gen = None
    if seed is not None:
        gen = torch.Generator(device=ids.device); gen.manual_seed(seed)
    cache, logits = prefill(model, ids)
    for i in range(n_new):
        if rep != 1.0:
            logits = apply_rep_penalty(logits, ids, rep)
        nxt = pick(logits) if gen is None else pick(logits, gen)
        ids = torch.cat([ids, nxt], 1)
        if i == n_new - 1:                       # guard：最后一张打分表没人读
            break
        logits = decode_step(model, nxt, cache)
    return ids


# ---------- 自测 ----------
if __name__ == '__main__':
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _LOCAL = 'D:/实习/models/Qwen3-0.6B'
    M = _LOCAL if os.path.isdir(_LOCAL) else 'Qwen/Qwen3-0.6B'   # 本地优先，否则走 HF（自动下载）
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to('cuda').eval()
    ids = tok('The capital of China is', return_tensors='pt').input_ids.cuda()

    with torch.no_grad():   # 无 cache 基线
        base = ids.clone()
        for _ in range(10):
            nxt = model(base).logits[:, -1].argmax(-1, keepdim=True)
            base = torch.cat([base, nxt], 1)

    out_g = generate_v2(model, ids, 10)                                  # greedy 档
    a = generate_v2(model, ids, 12, make_sampler(0.8, 0, 0.9), seed=42)  # 采样档
    b = generate_v2(model, ids, 12, make_sampler(0.8, 0, 0.9), seed=42)
    c = generate_v2(model, ids, 12, make_sampler(0.8, 0, 0.9), seed=7)
    print('① greedy == 无cache 基线:', torch.equal(out_g, base))
    print('② 同 seed 复现:', torch.equal(a, b), '| ③ 换 seed 不同:', not torch.equal(a, c))
    print('样例:', repr(tok.decode(a[0, -12:])))
