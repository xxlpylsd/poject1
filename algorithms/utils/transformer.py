from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt
from .util import init

def remap_legacy_state_dict(state_dict):
    out = {}
    for (k, v) in state_dict.items():
        k2 = k
        if k2.startswith('encoder.layers.'):
            k2 = k2[len('encoder.'):]
        elif '.encoder.layers.' in k2:
            k2 = k2.replace('.encoder.layers.', '.layers.')
        if k2.startswith('encoder.norm.'):
            k2 = 'norm_out.' + k2[len('encoder.norm.'):]
        elif '.encoder.norm.' in k2:
            k2 = k2.replace('.encoder.norm.', '.norm_out.')
        out[k2] = v
    return out

class TransformerEncoderBackbone(nn.Module):

    def __init__(self, input_dim: int, hidden_size: int, token_dim: int=1, d_model: int=64, n_head: int=4, n_layers: int=2, ff_mult: int=4, dropout: float=0.0, pool: str='mean', use_cls: bool=False, use_pe: bool=True, use_orthogonal: bool=True, use_ReLU: bool=True, max_tokens: int=256, grad_checkpoint: bool=False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.token_dim = max(1, int(token_dim))
        self.n_tokens = int(math.ceil(self.input_dim / self.token_dim))
        if self.n_tokens > max_tokens:
            raise ValueError(f'token 数 {self.n_tokens} 超过上限 {max_tokens}；请检查 obs_dim({self.input_dim}) / token_dim({self.token_dim}) 的设置。')
        self.padded_dim = self.n_tokens * self.token_dim
        self.d_model = int(d_model)
        self.pool = pool
        self.use_cls = bool(use_cls)
        self.use_pe = bool(use_pe)
        self.grad_checkpoint = bool(grad_checkpoint)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        activation = ['gelu', 'relu'][use_ReLU]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.input_proj = init_(nn.Linear(self.token_dim, self.d_model))
        n_pos = self.n_tokens + (1 if self.use_cls else 0)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        if self.use_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_pos, self.d_model))
            nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=int(n_head), dim_feedforward=int(d_model) * int(ff_mult), dropout=float(dropout), activation=activation, batch_first=True, norm_first=True)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(int(n_layers))])
        self.norm_out = nn.LayerNorm(self.d_model)
        self.out_proj = init_(nn.Linear(self.d_model, self.hidden_size))
        self._init_encoder_params(init_method)

    def _init_encoder_params(self, init_method):
        for (name, p) in self.named_parameters():
            if p.dim() > 1 and ('linear' in name or name.endswith('.weight')):
                if 'pos_embed' in name or 'cls_token' in name:
                    continue
                if 'in_proj_weight' in name or 'out_proj.weight' in name:
                    try:
                        init_method(p)
                    except Exception:
                        pass
                elif 'weight' in name and p.dim() == 2:
                    try:
                        init_method(p)
                    except Exception:
                        pass
            elif name.endswith('.bias'):
                nn.init.constant_(p, 0.0)

    def load_state_dict(self, state_dict, strict: bool=True, assign: bool=False):
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError:
            remapped = remap_legacy_state_dict(state_dict)
            return super().load_state_dict(remapped, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (pooled, _) = self.forward_with_tokens(x)
        return pooled

    def forward_with_tokens(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.padded_dim != self.input_dim:
            x = torch.cat([x, x.new_zeros(x.size(0), self.padded_dim - self.input_dim)], dim=-1)
        tokens = x.view(x.size(0), self.n_tokens, self.token_dim)
        h = self.input_proj(tokens)
        if self.use_cls:
            cls = self.cls_token.expand(h.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)
        if self.use_pe:
            h = h + self.pos_embed[:, :h.size(1), :]
        h = self._encode(h)
        if self.pool == 'cls' and self.use_cls:
            pooled = h[:, 0, :]
        elif self.pool == 'max':
            pooled = h.max(dim=1).values
        else:
            pooled = h.mean(dim=1)
        return (self.out_proj(pooled), h)

    def _encode(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                h = _ckpt(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return self.norm_out(h)

class TransformerBase(nn.Module):

    def __init__(self, args, obs_shape, token_dim: int=1):
        super().__init__()
        self._use_feature_normalization = getattr(args, 'use_feature_normalization', True)
        obs_dim = int(obs_shape[0])
        self.token_dim = max(1, int(token_dim))
        if obs_dim % self.token_dim != 0:
            raise ValueError(f'Transformer observation dimension must be divisible by token_dim: obs_dim={obs_dim}, token_dim={self.token_dim}')
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(self.token_dim)
        self.backbone = TransformerEncoderBackbone(input_dim=obs_dim, hidden_size=int(args.hidden_size), token_dim=int(token_dim), d_model=int(getattr(args, 'tf_d_model', 64)), n_head=int(getattr(args, 'tf_n_head', 4)), n_layers=int(getattr(args, 'tf_n_layers', 2)), ff_mult=int(getattr(args, 'tf_ff_mult', 4)), dropout=float(getattr(args, 'tf_dropout', 0.0)), pool=str(getattr(args, 'tf_pool', 'mean')), use_cls=bool(getattr(args, 'tf_use_cls', False)), use_pe=bool(getattr(args, 'tf_use_pe', True)), use_orthogonal=bool(getattr(args, 'use_orthogonal', True)), use_ReLU=bool(getattr(args, 'use_ReLU', True)), max_tokens=int(getattr(args, 'tf_max_tokens', 256)), grad_checkpoint=bool(getattr(args, 'tf_grad_checkpoint', False)))

    def forward(self, x):
        if self._use_feature_normalization:
            n_tokens = x.size(-1) // self.token_dim
            x = self.feature_norm(x.view(x.size(0), n_tokens, self.token_dim)).view(x.size(0), -1)
        return self.backbone(x)

    def forward_with_tokens(self, x):
        if self._use_feature_normalization:
            n_tokens = x.size(-1) // self.token_dim
            x = self.feature_norm(x.view(x.size(0), n_tokens, self.token_dim)).view(x.size(0), -1)
        (pooled, h) = self.backbone.forward_with_tokens(x)
        if self.backbone.use_cls:
            h = h[:, 1:, :]
        return (pooled, h)

    @property
    def has_tokens(self) -> bool:
        return True

def count_parameters(module: nn.Module) -> int:
    return int(sum((p.numel() for p in module.parameters() if p.requires_grad)))
