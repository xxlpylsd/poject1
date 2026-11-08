# -*- coding: utf-8 -*-
"""
轻量信道 API：生成/保存/加载 场景（AP/UE 位置、回程容量、大尺度衰落 mu）
供：界面A生成；基线/MAPPO评估复用；环境也可直接 load。
"""
from __future__ import annotations
import os, json, math
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional
import numpy as np

@dataclass
class ChannelConfig:
    M: int = 30              # UE 数
    N: int = 30              # AP 数
    map_size: float = 1000.0 # 地图边长(米)
    ap_min_distance: float = 0.0
    backhaul_cap_range: Tuple[float,float] = (4e8, 5e8)
    seed: int = 2

def _rng(seed:int): return np.random.default_rng(seed)

def _poisson_hardcore_aps(rng, N, L, dmin):
    """最小间距 dmin 的AP放置（简单拒绝采样，N不大时够用）"""
    if dmin <= 0: return rng.random((N,2))*L
    pts = []
    max_tries = 100000
    while len(pts) < N and max_tries>0:
        max_tries -= 1
        p = rng.random(2)*L
        if all(np.linalg.norm(p-q)>=dmin for q in pts):
            pts.append(p)
    if len(pts)<N:
        # 放宽一点，防止卡死
        return rng.random((N,2))*L
    return np.array(pts)

def _shadow_and_pathloss(rng, dist):
    M,N = dist.shape
    a=rng.random((1,N)); b=rng.random((1,M)); delta=rng.random()
    theta = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            theta[i,j] = math.sqrt(delta)*a[0,j] + math.sqrt(1-delta)*b[0,i]
    f=1.9*(10**3); h_ap,h_u=15,1.65
    L=46.3 + 33.9*np.log10(f) - 13.82*np.log10(h_ap) - (1.11*np.log10(f)-0.7)*h_u + 1.56*np.log10(f) - 0.8
    d0,d1=10.0,50.0
    mu_db = np.zeros_like(dist)
    for i in range(M):
        for j in range(N):
            d=dist[i,j]
            if d>d1:         mu_db[i,j] = -L - 35*np.log10(d/1000.0) + theta[i,j]
            elif d0<=d<=d1:  mu_db[i,j] = -L - 10*np.log10(d1**1.5 * (d)/(1000.0**2))
            else:            mu_db[i,j] = -L - 10*np.log10(d1**1.5 * d0**2)
    return 10**(mu_db/10.0)

def generate_channel(cfg: ChannelConfig) -> Dict[str, Any]:
    rng = _rng(cfg.seed)
    users = rng.random((cfg.M,2)) * cfg.map_size
    aps   = _poisson_hardcore_aps(rng, cfg.N, cfg.map_size, cfg.ap_min_distance)
    caps  = rng.uniform(cfg.backhaul_cap_range[0], cfg.backhaul_cap_range[1], size=cfg.N)
    dist = np.sqrt(((users[:,None,:]-aps[None,:,:])**2).sum(axis=2))
    mu   = _shadow_and_pathloss(rng, dist)
    return {
        "M": cfg.M, "N": cfg.N, "map_size": cfg.map_size,
        "ap_min_distance": cfg.ap_min_distance, "seed": cfg.seed,
        "users": users, "aps": aps, "caps": caps, "mu": mu
    }

def save_channel(scn: Dict[str,Any], path:str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **scn)
    return path

def load_channel(path:str) -> Dict[str,Any]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    # 兼容 JSON
    if "users" not in out and path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        for k,v in j.items():
            out[k] = np.array(v) if isinstance(v, list) else v
    return out
