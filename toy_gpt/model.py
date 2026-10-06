# -*- coding: utf-8 -*-
"""toy_gpt/model.py —— 0.21M 字符级 GPT（项目被测模型，Day 2 自研）
Head/MHA/FFN/Block/GPT 与训练脚本同构；Block 支持 ln_mode(two/shared/entrance)
与 norm_pos(pre/post) 以支撑架构对照实验（见 experiments/）。
参数量手推对账：209,729 ≈ 12·d²（emb 65×64 + 4 层 + pos 表 + lnf/head）。"""
import torch, torch.nn as nn, torch.nn.functional as F


class Head(nn.Module):
    """单头 self-attention：打分 ÷√d → 因果掩码 → softmax → 加权 V。"""
    def __init__(self, n_embd, hs, block_size):
        super().__init__(); self.hs = hs
        self.key   = nn.Linear(n_embd, hs, bias=False)
        self.query = nn.Linear(n_embd, hs, bias=False)
        self.value = nn.Linear(n_embd, hs, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))   # 下三角=因果掩码
    def forward(self, x):                                    # x: (B, T, 64)
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)                   # 各 (B, T, hs=16)
        wei = q @ k.transpose(-2, -1) * self.hs ** -0.5      # 打分表 (B, T, T)：每位置给全部历史打分
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # 挡未来：只许读 ≤自己 的位置
        wei = F.softmax(wei, dim=-1)                         # 分数→权重（每行和=1）
        return wei @ self.value(x)                           # (B, T, hs)：加权 V——attention 唯一"取信息"的一步


class MHA(nn.Module):
    """多头：h 份 Head 各算各的 → cat 只并排粘贴 → proj 压回 + 融合（唯一跨头混合处）。"""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // n_head, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FFN(nn.Module):
    """逐位置独立的两层 MLP，中间宽 4d：思考容量的大头（参数占 8d²/12d²）。"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd))
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    """残差流上的两个车间：attention（跨位置通信）+ FFN（逐位置思考），pre-norm 保护流量。
    ln_mode: 'two'(标准) / 'shared' / 'entrance'；norm_pos: 'pre' / 'post'
    （两种开关只为对照实验存在，见 experiments/exp2、exp3）"""
    def __init__(self, n_embd, n_head, block_size, ln_mode='two', norm_pos='pre'):
        super().__init__()
        assert not (norm_pos == 'post' and ln_mode != 'two')
        self.ln_mode, self.norm_pos = ln_mode, norm_pos
        self.sa = MHA(n_embd, n_head, block_size); self.ffwd = FFN(n_embd)
        if ln_mode == 'shared':
            self.ln = nn.LayerNorm(n_embd)
        else:
            self.ln1 = nn.LayerNorm(n_embd)
            if ln_mode == 'two':
                self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        if self.norm_pos == 'pre':
            if self.ln_mode == 'two':
                x = x + self.sa(self.ln1(x)); x = x + self.ffwd(self.ln2(x))     # 先标准化再进车间，增量加回干道
            elif self.ln_mode == 'shared':
                x = x + self.sa(self.ln(x));   x = x + self.ffwd(self.ln(x))
            else:
                x = x + self.sa(self.ln1(x)); x = x + self.ffwd(x)
        else:
            x = self.ln1(x + self.sa(x));     x = self.ln2(x + self.ffwd(x))     # post-norm：先加工再标准化
        return x


class GPT(nn.Module):
    """五零件：查身份(tok) + 盖位置(pos) → Block×n 同形加工 → lnf 出厂质检 → head 打分。"""
    def __init__(self, vocab_size, n_layer=4, n_embd=64, n_head=4, block_size=32,
                 ln_mode='two', norm_pos='pre'):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)          # 可学习绝对位置表（Qwen3 换成 RoPE，零参数）
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, ln_mode, norm_pos)
                                      for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))   # (B,T,64)：带住址的名片
        x = self.blocks(x); x = self.lnf(x)
        logits = self.head(x)                                # (B, T, V)：每个位置一张打分表
        loss = None if targets is None else \
            F.cross_entropy(logits.view(B*T, -1), targets.reshape(B*T))    # 训练：全 T 位置一次结算
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n_tok, block_size=32):
        for _ in range(n_tok):
            logits, _ = self(idx[:, -block_size:])           # 滑窗：最多回看 block_size（pos 表上限）
            probs = F.softmax(logits[:, -1, :], dim=-1)      # 只读最后一张打分表
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
