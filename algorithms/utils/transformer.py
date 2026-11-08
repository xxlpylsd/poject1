# -*- coding: utf-8 -*-
"""
algorithms/utils/transformer.py —— 用 Transformer 编码器替换 MAPPO 的 MLP 特征提取骨干

对应 想法(1).docx 的要求：

    观测 o_k
      -> 输入投影 Linear（把 token 维度映射到 d_model）
      -> 位置编码
      -> N 层 Transformer Encoder（LayerNorm + 多头自注意力 + 残差 + FFN + 残差）
      -> Actor 头：Linear + Softmax
         Critic 头：Linear -> 标量 V

要点（和想法文档里强调的一致）：
  * 被替换的是"隐藏层特征提取器"，输入投影（token -> d_model）与输出头仍然是全连接层；
  * Transformer 内部本身也由全连接层构成（Q/K/V 投影、注意力的输出投影、FFN 两层）；
  * 注意力需要"序列维"，所以先把观测切分成若干 token：
        - Actor 侧（[C1] AP-token）：观测被组织成 (1 + Kmax) 个 token，第 0 个是自身状态
          token，其余是 Kmax 个 AP token。注意力在 **AP 维**上做：决定该关注哪几个 AP。
        - Critic 侧（多智能体注意力，对应想法文档第 4 条）：全局状态是 M 个 UE 观测的拼接，
          天然就是 M 个"agent token"的序列。注意力在 **UE 维**上做：决定该关注哪些 UE 的
          状态来估计全局价值（谁在被谁干扰）。

实现上这是一个通用的 "tokenizer + encoder + pool" 模块：
    输入 (B, input_dim)  ->  内部 padding 到 (B, n_tokens, token_dim)
                        ->  Linear -> (+位置编码) -> N x Encoder -> 池化 -> Linear
所以它对 Actor / Critic、对不同的 obs 维度都能直接用（n_tokens 由 token_dim 自动推出）。
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt

from .util import init


def remap_legacy_state_dict(state_dict):
    """把"旧版 Transformer 骨干"的参数键名映射到当前实现。

    历史原因：为了支持梯度检查点，本文件把 `nn.TransformerEncoder`（属性名 encoder）
    手工展开成了 `self.layers` + `self.norm_out`，于是参数键名从
        base.backbone.encoder.layers.0.self_attn.in_proj_weight
        base.backbone.encoder.norm.weight
    变成
        base.backbone.layers.0.self_attn.in_proj_weight
        base.backbone.norm_out.weight
    两者在**数值上完全等价**（只是命名不同），所以这里做一次键名重映射，
    让梯度检查点改动之前训练出来的模型仍然可以被加载（否则补评测会直接报错）。
    """
    out = {}
    for k, v in state_dict.items():
        k2 = k
        if k2.startswith("encoder.layers."):          # 直接对 backbone 加载时
            k2 = k2[len("encoder."):]
        elif ".encoder.layers." in k2:                # 从 R_Actor/R_Critic 里加载时
            k2 = k2.replace(".encoder.layers.", ".layers.")
        if k2.startswith("encoder.norm."):
            k2 = "norm_out." + k2[len("encoder.norm."):]
        elif ".encoder.norm." in k2:
            k2 = k2.replace(".encoder.norm.", ".norm_out.")
        out[k2] = v
    return out


class TransformerEncoderBackbone(nn.Module):
    """把定长观测向量当作 token 序列，用 Transformer Encoder 提特征。"""

    def __init__(self,
                 input_dim: int,
                 hidden_size: int,
                 token_dim: int = 1,
                 d_model: int = 64,
                 n_head: int = 4,
                 n_layers: int = 2,
                 ff_mult: int = 4,
                 dropout: float = 0.0,
                 pool: str = "mean",
                 use_cls: bool = False,
                 use_pe: bool = True,
                 use_orthogonal: bool = True,
                 use_ReLU: bool = True,
                 max_tokens: int = 256,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.token_dim = max(1, int(token_dim))
        self.n_tokens = int(math.ceil(self.input_dim / self.token_dim))
        if self.n_tokens > max_tokens:
            raise ValueError(
                f"token 数 {self.n_tokens} 超过上限 {max_tokens}；"
                f"请检查 obs_dim({self.input_dim}) / token_dim({self.token_dim}) 的设置。"
            )
        self.padded_dim = self.n_tokens * self.token_dim
        self.d_model = int(d_model)
        self.pool = pool
        self.use_cls = bool(use_cls)
        self.use_pe = bool(use_pe)
        # 梯度检查点：反向时重算每层激活，显存降 3~4 倍（数学完全不变，只是慢约 20~30%）
        # batch 很大时（例如 T*N*M = 64*16*30 = 30720、48 个 token）必须开，否则 24GB 卡会 OOM
        self.grad_checkpoint = bool(grad_checkpoint)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        activation = ["gelu", "relu"][use_ReLU]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # 1) 输入投影：token_dim -> d_model（仍然是全连接层）
        self.input_proj = init_(nn.Linear(self.token_dim, self.d_model))

        # 2) [CLS] token（可选）与位置编码
        n_pos = self.n_tokens + (1 if self.use_cls else 0)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        if self.use_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_pos, self.d_model))
            nn.init.normal_(self.pos_embed, std=0.02)

        # 3) N 层 Transformer Encoder（pre-LN，训练更稳）
        # 注意：这里手工展开层循环（而不是用 nn.TransformerEncoder 包一层），
        # 是为了能对**每一层**做梯度检查点。
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_head),
            dim_feedforward=int(d_model) * int(ff_mult),
            dropout=float(dropout),
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(int(n_layers))])
        self.norm_out = nn.LayerNorm(self.d_model)

        # 4) 输出头：d_model -> hidden_size（仍然是全连接层）
        self.out_proj = init_(nn.Linear(self.d_model, self.hidden_size))

        # 初始化 Q/K/V 与 FFN 的线性层
        self._init_encoder_params(init_method)

    def _init_encoder_params(self, init_method):
        for name, p in self.named_parameters():
            if p.dim() > 1 and ("linear" in name or name.endswith(".weight")):
                if "pos_embed" in name or "cls_token" in name:
                    continue
                if "in_proj_weight" in name or "out_proj.weight" in name:
                    try:
                        init_method(p)
                    except Exception:
                        pass
                elif "weight" in name and p.dim() == 2:
                    try:
                        init_method(p)
                    except Exception:
                        pass
            elif name.endswith(".bias"):
                nn.init.constant_(p, 0.0)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """先按当前键名加载；失败则尝试旧版键名（兼容梯度检查点改动之前的 checkpoint）。"""
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError:
            remapped = remap_legacy_state_dict(state_dict)
            return super().load_state_dict(remapped, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.forward_with_tokens(x)
        return pooled

    def forward_with_tokens(self, x: torch.Tensor):
        """同时返回池化特征与整条 token 序列（供"逐 token 打分"的集合动作头使用）。

        返回：(pooled [B,hidden], h [B, T(+1), d_model])
        """
        # x: (B, input_dim) -> (B, n_tokens, token_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.padded_dim != self.input_dim:
            x = torch.cat([x, x.new_zeros(x.size(0), self.padded_dim - self.input_dim)], dim=-1)
        tokens = x.view(x.size(0), self.n_tokens, self.token_dim)

        h = self.input_proj(tokens)                       # (B, T, d_model)
        if self.use_cls:
            cls = self.cls_token.expand(h.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)
        if self.use_pe:
            h = h + self.pos_embed[:, : h.size(1), :]

        h = self._encode(h)                               # (B, T(+1), d_model)

        if self.pool == "cls" and self.use_cls:
            pooled = h[:, 0, :]
        elif self.pool == "max":
            pooled = h.max(dim=1).values
        else:
            pooled = h.mean(dim=1)
        return self.out_proj(pooled), h

    def _encode(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                h = _ckpt(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return self.norm_out(h)


class TransformerBase(nn.Module):
    """与 MLPBase 同接口的封装，供 R_Actor / R_Critic 直接替换使用。"""

    def __init__(self, args, obs_shape, token_dim: int = 1):
        super().__init__()
        self._use_feature_normalization = getattr(args, "use_feature_normalization", True)
        obs_dim = int(obs_shape[0])
        self.token_dim = max(1, int(token_dim))
        if obs_dim % self.token_dim != 0:
            raise ValueError(
                "Transformer observation dimension must be divisible by "
                f"token_dim: obs_dim={obs_dim}, token_dim={self.token_dim}"
            )
        if self._use_feature_normalization:
            # Normalize each structured token independently. A LayerNorm over
            # the flattened observation would mix the self token with AP
            # tokens and weaken permutation behaviour before attention.
            self.feature_norm = nn.LayerNorm(self.token_dim)

        self.backbone = TransformerEncoderBackbone(
            input_dim=obs_dim,
            hidden_size=int(args.hidden_size),
            token_dim=int(token_dim),
            d_model=int(getattr(args, "tf_d_model", 64)),
            n_head=int(getattr(args, "tf_n_head", 4)),
            n_layers=int(getattr(args, "tf_n_layers", 2)),
            ff_mult=int(getattr(args, "tf_ff_mult", 4)),
            dropout=float(getattr(args, "tf_dropout", 0.0)),
            pool=str(getattr(args, "tf_pool", "mean")),
            use_cls=bool(getattr(args, "tf_use_cls", False)),
            use_pe=bool(getattr(args, "tf_use_pe", True)),
            use_orthogonal=bool(getattr(args, "use_orthogonal", True)),
            use_ReLU=bool(getattr(args, "use_ReLU", True)),
            max_tokens=int(getattr(args, "tf_max_tokens", 256)),
            grad_checkpoint=bool(getattr(args, "tf_grad_checkpoint", False)),
        )

    def forward(self, x):
        if self._use_feature_normalization:
            n_tokens = x.size(-1) // self.token_dim
            x = self.feature_norm(
                x.view(x.size(0), n_tokens, self.token_dim)
            ).view(x.size(0), -1)
        return self.backbone(x)

    def forward_with_tokens(self, x):
        """返回 (池化特征, token 序列)。token 序列已包含可选的 [CLS] 前缀。"""
        if self._use_feature_normalization:
            n_tokens = x.size(-1) // self.token_dim
            x = self.feature_norm(
                x.view(x.size(0), n_tokens, self.token_dim)
            ).view(x.size(0), -1)
        pooled, h = self.backbone.forward_with_tokens(x)
        if self.backbone.use_cls:
            h = h[:, 1:, :]        # 去掉 [CLS]，让下标与输入 token 一一对应
        return pooled, h

    @property
    def has_tokens(self) -> bool:
        return True


def count_parameters(module: nn.Module) -> int:
    """可训练参数量（用于"两个模型容量是否可比"的体检）。"""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
