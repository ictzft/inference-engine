# -*- coding: utf-8 -*-
"""demo.py —— 一行命令看引擎效果（v2 两阶段 + 可插拔采样）

用法（在仓库根目录下）：
    python demo.py                                          # 裸续写：默认英文 prompt，T=0.8 / top-p=0.9
    python demo.py "从前有一座山，山里有座庙，" --n 60        # 自定义 prompt + 长度
    python demo.py "用一句话解释什么是 KV Cache" --chat     # 对话模式：chat template + EOS 自然停
    python demo.py "中国的首都是哪里？" --chat --seed 42    # seed 给定 → 同命令两遍逐字复现

完整参数说明：python demo.py --help
"""
import argparse, os, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from engine.v2_two_stage import generate_v2, make_sampler   # v2 引擎：两阶段生成 + 采样组合器


def main():
    # ---------- 命令行接口：把 "--n 80 --t 0.3" 这串字翻译成 args.n / args.t ----------
    ap = argparse.ArgumentParser(description='v2 两阶段引擎演示')
    ap.add_argument('prompt', nargs='?', default='The capital of China is')  # 位置参数，可省
    ap.add_argument('--n', type=int, default=50, help='生成 token 数')       # 防爆上限；--chat 下由 EOS 决定实际停点
    ap.add_argument('--t', type=float, default=0.8, help='temperature')      # 低温=固定，高温=发散
    ap.add_argument('--p', type=float, default=0.9, help='top-p')            # 累计概率截断
    ap.add_argument('--seed', type=int, default=None, help='给定则逐字可复现')  # 不给=公共随机源，不可复现
    ap.add_argument('--chat', action='store_true',
                    help='套 Qwen3 对话模板（enable_thinking=False）+ EOS 自然停')  # 开关型：后面不跟值
    args = ap.parse_args()

    # ---------- 模型加载：本地路径优先，不存在回退 HF id（他人 clone 后自动下载约 1.2GB） ----------
    _LOCAL = 'D:/实习/models/Qwen3-0.6B'
    M = _LOCAL if os.path.isdir(_LOCAL) else 'Qwen/Qwen3-0.6B'
    tok = AutoTokenizer.from_pretrained(M)                                    # 调度员：str ↔ token id
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to('cuda').eval()

    # ---------- 进料 + 停法：--chat 是一把双掷开关（换制服，也换终点） ----------
    eos = None
    if args.chat:
        # 对话模板：裸续写 → 问答任务（「格式即指令」）——原话原封嵌在对话框架中间
        #   add_generation_prompt=True：尾部接 assistant 开头（把话筒递给助手角色）
        #   enable_thinking=False：Qwen3 专属开关，写一个空思考块占位
        # v5.17 实测：返回 BatchEncoding，input_ids 已是 (1,T) 张量
        enc = tok.apply_chat_template([{'role': 'user', 'content': args.prompt}],
                                      tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False, return_tensors='pt')
        ids = enc.input_ids.cuda()                    # (1, T)：T = 对话框架 + 原话
        eos = tok.eos_token_id                        # 听见就停：不跑满 n，且不入历史不入输出
    else:
        # 裸续写：不带任何框架，模型「接着写文档」；没有说完的约定 → 跑满 n 个
        ids = tok(args.prompt, return_tensors='pt').input_ids.cuda()          # (1, T)

    # ---------- 采样策略：组合器造 pick 闭包（温度 → top-k 关 → top-p → 抽签） ----------
    pick = make_sampler(args.t, 0, args.p)            # 进 (1,V) 打分表 → 出 (1,1) token，可插拔

    # ---------- 计时协议：synchronize 前后夹住（CUDA 异步——不同步，量到的是「递交时间」=假快） ----------
    plen = ids.shape[1]                               # prompt 长度，之后只取生成段
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = generate_v2(model, ids, args.n, pick, seed=args.seed, eos=eos)      # (1, plen+已生成)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # ---------- 出字：chat 只打回答段（模板脚手架不显示）；裸续写 prompt+续写一起看 ----------
    if args.chat:
        print(tok.decode(out[0, plen:]))
    else:
        print(tok.decode(out[0]))
    n_new = out.shape[1] - plen                       # 实际生成数（--chat 自然停会 < n）
    print(f'\n[{n_new} tokens / {dt:.2f}s = {n_new/dt:.1f} tok/s（含 TTFT）；'
          f'T={args.t} top_p={args.p} seed={args.seed} chat={args.chat}]')


if __name__ == '__main__':      # 被直接运行才执行 main；被 import 时静默（Python 标配守卫）
    main()
