# -*- coding: utf-8 -*-
"""demo.py —— 一行命令看引擎效果（v2 两阶段 + 可插拔采样）

用法（在仓库根目录下）：
    python demo.py                                # 默认英文 prompt，T=0.8 top-p=0.9
    python demo.py "中国的首都是" --n 50 --t 0.8 --p 0.9 --seed 42
"""
import argparse, os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from engine.v2_two_stage import generate_v2, make_sampler


def main():
    ap = argparse.ArgumentParser(description='v2 两阶段引擎演示')
    ap.add_argument('prompt', nargs='?', default='The capital of China is')
    ap.add_argument('--n', type=int, default=50, help='生成 token 数')
    ap.add_argument('--t', type=float, default=0.8, help='temperature')
    ap.add_argument('--p', type=float, default=0.9, help='top-p')
    ap.add_argument('--seed', type=int, default=None, help='给定则逐字可复现')
    args = ap.parse_args()

    _LOCAL = 'D:/实习/models/Qwen3-0.6B'
    M = _LOCAL if os.path.isdir(_LOCAL) else 'Qwen/Qwen3-0.6B'   # 本地优先，否则走 HF
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to('cuda').eval()

    ids = tok(args.prompt, return_tensors='pt').input_ids.cuda()
    pick = make_sampler(args.t, 0, args.p)

    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = generate_v2(model, ids, args.n, pick, seed=args.seed)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    print(tok.decode(out[0]))
    n_new = out.shape[1] - ids.shape[1]
    print(f'\n[{n_new} tokens / {dt:.2f}s = {n_new/dt:.1f} tok/s（含 TTFT）；'
          f'T={args.t} top_p={args.p} seed={args.seed}]')


if __name__ == '__main__':
    main()
