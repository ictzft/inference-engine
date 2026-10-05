# -*- coding: utf-8 -*-
"""toy_gpt/train.py —— 训练入口（5070 Ti 实测：val 1.76 @ 5000 步约 108s）
用法：python train.py [iters]    # 默认 5000；quick=200 自检"""
import sys, time
from pathlib import Path
import torch
from model import GPT

BASE = Path(__file__).resolve().parent
text = open(BASE / 'input.txt', encoding='utf-8').read()
chars = sorted(set(text)); vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size, block_size = 32, 32
max_iters  = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
lr = 1e-3

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate(m, iters=200):
    out = {}
    for sp in ['train', 'val']:
        ls = torch.zeros(iters)
        for k in range(iters):
            X, Y = get_batch(sp); _, l = m(X, Y); ls[k] = l.item()
        out[sp] = ls.mean()
    return out

def main():
    torch.manual_seed(1337)
    m = GPT(vocab_size, n_layer=4, n_embd=64, n_head=4, block_size=block_size).to(device)
    print('参数量:', sum(p.numel() for p in m.parameters()))
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    t0 = time.time()
    for it in range(max_iters):
        if (it + 1) % 1000 == 0 or it == 0:
            L = estimate(m, iters=min(200, max_iters))
            print(f'step {it+1}: train {L["train"]:.4f}, val {L["val"]:.4f}')
        xb, yb = get_batch('train')
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    print(f'耗时 {time.time()-t0:.0f}s')
    torch.save(m.state_dict(), BASE / 'toy_gpt.pt')

if __name__ == '__main__':
    main()
