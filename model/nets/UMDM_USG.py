# This file is modified from https://github.com/NVlabs/Sana

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma

import os
import torch
import torch.nn as nn
import random
from timm.models.layers import DropPath

from model.builder import MODELS
from model.nets.modules import GLUMBConv
from model.nets.blocks import (
    Attention,
    CaptionEmbedder,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    PatchEmbedMS,
    T2IFinalLayer,
    TimestepEmbedder,
    t2i_modulate,
    get_2d_sincos_pos_embed,
)
from model.nets.norms import RMSNorm
from model.utils import auto_grad_checkpoint
from utils.dist_utils import get_rank
from utils.import_utils import is_xformers_available
from utils.logger import get_root_logger

_xformers_available = False
if is_xformers_available():
    _xformers_available = True


# LayerNorm + Attention + MLP
class MAM(nn.Module):
    """
    A Transformer block with global shared adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        qk_norm=False,
        attn_type="linear",
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if attn_type == "flash":  # flash self attention
            self.attn = FlashAttention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm, **block_kwargs)
        elif attn_type == "vanilla":  # vanilla self attention
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        elif attn_type == "linear":  # linear self attention
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs)

        self.attn2 = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = GLUMBConv(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=mlp_acts,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)
        self.gate_proj = nn.Linear(hidden_size, 2, bias=True)

    # x图像 y文本 t时间 r条件 role[0 1 2]
    def forward(self, x, y, t, r, role, mask=None, HW=None, **kwargs):
        B, L, N, C = x.shape  # L = 1 + num_conditions
        assert t.shape == (B, 6 * C), f"{t.shape} != {(B, 6 * C)}"          # timestep 条件（6倍展开）每层 AdaLN 有 6个参数
        assert r.shape == (B, L, 6 * C), f"{r.shape} != {(B, L, 6 * C)}"    # 也是 6 个参数
        assert role.shape == (B, L), f"{role.shape} != {(B, L)}"

        # AdaLN 条件参数 AdaLN-Zero
        # 是一种将外部条件（如文本、时间步、类别、图像等）注入到 Transformer 模型内部的方式
        # 通过动态生成 LayerNorm 的 scale 和 shift，以灵活控制模型行为
        # （scale）和（shift）替换为从条件中计算得到的动态向量
        # 就是这个东西 来实现生成多个结果的
        adaln_params = [
            (self.scale_shift_table[None] + t.reshape(B, 6, -1) + r[:, i].reshape(B, 6, -1)).chunk(6, dim=1)
            for i in range(L)
        ]
        # adaln_params[i] = [
        #     scale_1, shift_1,    # Self-Attn 前的 AdaLN
        #     scale_2, shift_2,    # FFN 前的 AdaLN
        #     scale_3,             # Self-Attn 后残差调节
        #     scale_4,             # FFN 后残差调节
        # ]
        # 要放在这6个地方，所有上面都是6 让每个位置都可以根据条件（文本、时间、角色）独立控制，同时保证模型在训练初期不破坏已有结构
        # 为什么是6个呢，就是原本进行层归一化的地方换成了AdaLN 而交叉注意力这只是和文本建立联系，就不需要了


        # Self attention （分图处理 → 合并 Attention → 再分开） 把所有的拼在一起做统一的自注意力 这样不同图之间可以信息交互
        # modulate seperately
        # 对每张图 x[:, i] 做 LayerNorm 再用对应的 scale + shift（来自 adaln）调节
        
        # -------- 预归一化 + 调制（逐模态）--------
        z_list = [t2i_modulate(self.norm1(x[:, i]), adaln_params[i][0], adaln_params[i][1]) for i in range(L)]  # B,N,C

        # -------- 分支B：全局自注意力（保持原逻辑）--------
        ignore_global = torch.eq(role, 2).repeat_interleave(N, dim=1)  # (B, L*N)
        z_all = torch.cat(z_list, dim=1)                               # (B, L*N, C)
        z_global = self.attn(z_all, HW=HW, ignore=ignore_global, block_id=kwargs.get("block_id", None))  # (B, L*N, C)

        # -------- 分支A：中心(0) <-> 条件(1) 双向 cross-attn 对齐 --------
        # 先准备一个承载张量
        z_pair = torch.zeros_like(z_all)

        # 每个样本独立处理（role 掩码因 batch 样本不同而不同）
        for b in range(B):
            role_b = role[b]              # (L,)
            idx_center = (role_b == 0).nonzero(as_tuple=False).flatten()
            idx_cond   = (role_b == 1).nonzero(as_tuple=False).flatten()

            if idx_center.numel() == 0 or idx_cond.numel() == 0:
                # print(f"Sample {b}: num_role0={idx_center.numel()}, num_role1={idx_cond.numel()}")
                dummy_q = torch.randn((1, max(1, N), C), device=x.device)
                dummy_kv = torch.randn((1, max(1, N), C), device=x.device)
                z_pair = z_pair + 0.0 * self.attn2(dummy_q, dummy_kv).sum(dim=(1,2), keepdim=True)
                continue  # 没有中心或没有条件，跳过 pair 分支

            # print(f"Sample {b}: num_role0={idx_center.numel()}, num_role1={idx_cond.numel()}")
            # gather：把多中心/多条件各自拼接成一条序列
            q_center = torch.cat([z_list[i][b] for i in idx_center], dim=0)[None, ...]  # (1, Nc*N, C)
            kv_cond  = torch.cat([z_list[i][b] for i in idx_cond],   dim=0)[None, ...]  # (1, Nd*N, C)

            # center <- cond
            up_c = self.attn2(q_center, kv_cond, mask=None)      # (1, Nc*N, C)
            # cond   <- center
            up_d = self.attn2(kv_cond,  q_center, mask=None)     # (1, Nd*N, C)

            # scatter 回各自模态块在大序列中的位置
            # center 模态回填
            start = 0
            for i in idx_center.tolist():
                z_pair[b, i * N:(i + 1) * N] = up_c[0, start:start + N]
                start += N
            # cond 模态回填
            start = 0
            for i in idx_cond.tolist():
                z_pair[b, i * N:(i + 1) * N] = up_d[0, start:start + N]
                start += N

        # -------- 门控融合（逐模态 -> 逐 token 展开）--------
        # g_i = sigmoid(W * (scale_1_i + shift_1_i))  -> [g_pair, g_global]
        gates = []
        for i in range(L):
            ctx = (adaln_params[i][0] + adaln_params[i][1])  # (B,1,C)
            gi  = torch.sigmoid(self.gate_proj(ctx.squeeze(1)))  # (B,2)
            gates.append(gi)
        gates = torch.stack(gates, dim=1)  # (B, L, 2)
        g_pair   = gates[..., 0:1]         # (B, L, 1)
        g_global = gates[..., 1:2]         # (B, L, 1)

        # 若该样本无中心或无条件，则把 g_pair 置 0（避免把空结果混进去）
        has_center = (role == 0).any(dim=1, keepdim=True)  # (B,1)
        has_cond   = (role == 1).any(dim=1, keepdim=True)  # (B,1)
        has_pair   = (has_center & has_cond).float()       # (B,1)
        g_pair = g_pair * has_pair[:, None, :]             # (B,L,1)

        # 展开到 token 维度
        def expand_tokens(g_mod):  # (B,L,1) -> (B, L*N, 1)
            return g_mod.repeat_interleave(N, dim=1)

        gp_tok = expand_tokens(g_pair)
        gg_tok = expand_tokens(g_global)

        z_fused = gp_tok * z_pair + gg_tok * z_global  # (B, L*N, C)
        
        if random.random() < 0.001:
            with torch.no_grad():
                gp_mean = gp_tok.mean().item()
                gg_mean = gg_tok.mean().item()
                if get_rank() == 0:
                    print(f"[Gate @step] pair={gp_mean:.3f}, global={gg_mean:.3f}")

        

        # -------- 残差写回 & 文本 cross-attn（保持原逻辑）--------
        z_fused = z_fused.view(B, L, N, C)

        # 按图分开后，逐图乘以残差调节因子（AdaLN 第 3 个参数）
        z_res = [self.drop_path(adaln_params[i][2] * z_fused[:, i]) for i in range(L)]  # scale_3
        z_res = torch.stack(z_res, dim=1)
        x = x + z_res   # 残差

        # Cross attention 将所有图的 patch 合并 → 与文本 y 做 Cross-Attention 和文本对应
        x = x.reshape(B, L * N, C)  # (B, L * N, C)
        x = x + self.cross_attn(x, y, mask)
        x = x.reshape(B, L, N, C)  # (B, L, N, C)

        # Mix-FFN 对每张图做 LayerNorm + AdaLN 保留图像结构细节
        # modulate seperately
        z = [t2i_modulate(self.norm2(x[:, i]), adaln_params[i][3], adaln_params[i][4]) for i in range(L)]
        # feedforward separately
        z = [self.mlp(z[i], HW=HW) for i in range(L)]
        # modulate seperately
        # AdaLN
        z = [self.drop_path(adaln_params[i][5] * z[i]) for i in range(L)]
        z = torch.stack(z, dim=1)  # (B, L, N, C)
        x = x + z # 残差
        return x

        """
        MAM 前向流程（中心↔条件分支 + 全局对齐 + 文本对齐 + FFN 细化）

        Step 1: 模态内预处理（Norm + AdaLN 调制）
            - 对每个模态 x[:, i]:
                1. LayerNorm
                2. 根据时间步 t 和角色条件 r 进行 AdaLN 调制
            - 得到调制后的特征列表 z_list[i] (B, N, C)

        Step 2: 分支A（中心 <-> 条件 Cross-Attn 对齐）
            - 识别模态角色:
                role=0 → 中心模态（生成目标）
                role=1 → 条件模态（输入参考）
                role=2 → 空图（跳过）
            - 对每个 batch 样本:
                1. 找出中心模态序列 idx_center
                2. 找出条件模态序列 idx_cond
                3. 若同时存在:
                    - center <- cross_attn(center, cond)
                    - cond   <- cross_attn(cond, center) （梯度已 detach，不更新条件 encoder）
                4. 将更新结果回填到对应模态位置
            - 输出 z_pair（中心-条件对齐分支）

        Step 3: 分支B（全模态统一 Self-Attn 对齐）
            - 将所有模态 token 拼接成一条长序列
            - 根据 role 生成 ignore mask（跳过空图 role=2）
            - 输入 self.attn，得到全模态混合后的更新 z_global

        Step 4: 门控融合（逐模态 -> token）
            - 对每个模态计算两个门控权重：
                g_pair   → 分支A（中心-条件对齐）权重
                g_global → 分支B（全局对齐）权重
            - 若样本无中心或条件，则强制 g_pair=0
            - 将模态级权重扩展到 token 级别
            - 融合更新：
                z_all = g_pair * z_pair + g_global * z_global
                x = x + z_all

        Step 5: 文本 Cross-Attn
            - 将所有模态 token 拼接
            - 执行 cross_attn(x, y)（和文本 y 对齐）
            - 再 reshape 回多模态结构

        Step 6: 模态内 FFN 细化
            - 对每个模态：
                1. LayerNorm + AdaLN 调制
                2. 进入卷积 MLP
                3. 残差连接 + DropPath
            - 得到最终输出 x
        """

@MODELS.register_module()
class UMDM(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        drop_path=0.0,
        caption_channels=2304,
        pe_interpolation=1.0,
        config=None,
        model_max_length=300,
        qk_norm=False,
        y_norm=False,
        norm_eps=1e-5,
        attn_type="linear",
        use_pe=False,
        y_norm_scale_factor=1.0,
        patch_embed_kernel=None,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
        num_conditions=1,
        freeze_attn=True,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.class_dropout_prob = class_dropout_prob
        self.pe_interpolation = pe_interpolation
        self.use_pe = use_pe
        self.y_norm = y_norm
        self.fp32_attention = kwargs.get("use_fp32_attention", False)
        self.num_conditions = num_conditions
        self.h = self.w = 0

        # Patch embedding
        kernel_size = patch_embed_kernel or patch_size

        # x: 支持 1+K 个图（主图 + K 个条件图），每个图用一个 patch embedding
        self.x_embedders = nn.ModuleList([
            PatchEmbedMS(patch_size, in_channels, hidden_size, kernel_size=kernel_size, bias=True)
            for _ in range(1 + num_conditions)
        ])

        # Time embedding 把时间步编码成可用于 AdaLN 的条件向量 注意6个
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Position embedding (dynamically computed in forward pass)
        self.base_size = input_size // patch_size
        self.pos_embed_ms = None

        # Caption embedding
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        # 文本
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        if self.y_norm:
            self.attention_y_norm = RMSNorm(hidden_size, scale_factor=y_norm_scale_factor, eps=norm_eps)

        # Role embedding 每张图根据角色注入控制信息，参与 AdaLN
        self.role_embedder = nn.Embedding(3, hidden_size)  # 0: generated, 1: condition, 2: null
        self.role_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # Domain embedding 位置性标记，每张图一个 domain embedding（用于打破模态混淆）
        # 标记不同图像通道（主图、各类条件图）的 learnable embedding，提供“模态位置感知” 给出每张图的「模态/通道位置信息」
        # 表示是什么模态 来区分是seg depth normal 第几个条件图
        self.domain_embedding = nn.Parameter(torch.randn(1+num_conditions, hidden_size))

        # Transformer blocks 
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                MAM(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=qk_norm,
                    attn_type=attn_type,
                    mlp_acts=mlp_acts,
                    linear_head_dim=linear_head_dim,
                    cross_norm=cross_norm,
                )
                for i in range(depth)
            ]
        )
        
        if freeze_attn:
            for block in self.blocks:
                for p in block.attn.parameters():
                    p.requires_grad = False

            # for block in self.blocks:
                # block.gate_proj.bias.data = torch.tensor([1.0, -1.0])

        self.freeze_attn = freeze_attn
        
        
        #############################################
        self.projector = build_mlp(hidden_size)  # every img projector
        ###############################################

        # Final layer
        # 最后还原
        self.final_layers = nn.ModuleList([
            T2IFinalLayer(hidden_size, patch_size, self.out_channels)
            for _ in range(1 + num_conditions)
        ])

        # Weights initialization
        self.initialize()

        logger = get_root_logger(os.path.join(config.work_dir, "train_log.log")).warning if config else print
        if get_rank() == 0:
            logger(f"use pe: {use_pe}")
            logger(f"position embed interpolation: {self.pe_interpolation}")
            logger(f"base size: {self.base_size}")
            logger(f"attention type: {attn_type}")
            logger(f"autocast linear attn: {os.environ.get('AUTOCAST_LINEAR_ATTN', False)}")

    def forward(self, x, timestep, y, role, mask=None, clean_x=None, **kwargs):
        """
        Forward pass of UMDM.
        x: (N, 1+K, C, H, W) tensor of spatial inputs (latent representations of images and conditions)
        t: (N, ) tensor of diffusion timesteps
        y: (N, 1, 300, C) tensor of text embeddings
        role: (N, 1+K) tensor of role (0: generated, 1: condition, 2: null)
        clean_x: (N, 1+K, C, H, W) tensor of clean images (optional)  # 条件图原始 latent（供 role==1 替换）
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        role = role.to(torch.long)
        assert x.shape[1] == 1 + self.num_conditions, f"{x.shape[1]} != {1 + self.num_conditions}"
        assert role.shape[1] == 1 + self.num_conditions, f"{role.shape[1]} != {1 + self.num_conditions}"
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        if clean_x is None:
            clean_x = torch.zeros_like(x)

        # Handle cfg stacking: the sampler won't stack model_kwargs (role and clean_x)
        # TODO: should find a better way to handle this
        # Classifier-Free Guidance (CFG) 是一种不使用分类器的条件引导方法。
        # 通过同时输入有条件（如文本）和无条件（如空文本）的样本，来加强模型对条件的遵循程度。
        if role.shape[0] * 2 == x.shape[0]:
            role = role.repeat(2, 1)
        if clean_x.shape[0] * 2 == x.shape[0]:
            clean_x = clean_x.repeat(2, 1, 1, 1, 1)

        # Replace role==1 with clean_x
        # 条件图会用role==1来替代
        if torch.eq(role, 1).any():
            clean_x = clean_x.to(self.dtype)
            x = torch.where(torch.eq(role, 1)[..., None, None, None], clean_x, x)

        # Patch embedding
        x = torch.unbind(x, dim=1)  # tuple of (N, C, H, W)
        x = [x_embedder(x[i]) for i, x_embedder in enumerate(self.x_embedders)]
        x = torch.stack(x, dim=1)  # (N, 1+K, T, D)

        # Add positional embedding
        # 加位置编码 论文里说加不加无所谓
        if self.use_pe:
            if self.pos_embed_ms is None or self.pos_embed_ms.shape[1:] != x[0].shape[1:]:
                self.pos_embed_ms = (
                    torch.from_numpy(
                        get_2d_sincos_pos_embed(
                            self.hidden_size,
                            (self.h, self.w),
                            pe_interpolation=self.pe_interpolation,
                            base_size=self.base_size,
                        )
                    ).unsqueeze(0).to(x[0].device).to(self.dtype)
                )
            x = torch.unbind(x, dim=1)  # tuple of (N, T, D)
            x = [_x + self.pos_embed_ms for _x in x]
            x = torch.stack(x, dim=1)  # (N, 1+K, T, D)

        # Replace role==2 with zero
        x = torch.where(torch.eq(role, 2)[..., None, None], 0., x)

        # Role embedding + Domain embedding
        r = self.role_embedder(role)  # (N, 1+K, D)         我的身份信息
        r = r + self.domain_embedding[None]  # (N, 1+K, D)  这张图是谁（主图/第几个条件图）
        r0 = self.role_block(r)  # (N, 1+K, 6 * D)          做一次 MLP 映射成 AdaLN 的 6 个参数
        # 这是第 i 张图，它的任务角色是 role=i，现在我们给你一套参数，请用来调节这张图在模型里的作用。

        # Time embedding
        t = self.t_embedder(timestep)  # (N, D)
        t0 = self.t_block(t)  # (N, 6 * D)      # MLP，将其展开为 AdaLN-Zero 所需的 6 个向量

        # Caption embedding
        force_drop_ids = (role[:, 0] == 1)      # 若主图是条件图，就强制drop text

        # 每个样本以 p=0.1 的概率 drop 掉文本（训练 CFG 的无条件分支）
        if self.training and self.class_dropout_prob > 0:
            cfg_force_drop_ids = (torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob)     #  # drop 掉一部分 token（CFG）
            force_drop_ids = force_drop_ids | cfg_force_drop_ids
        # 将 [B, 1, L_text, D_in] 的文本 token 输入转换为 [B, 1, L_text, D]
        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)  # (N, 1, L, D)
        if self.y_norm:
            y = self.attention_y_norm(y)

        # 若开启了 CFG，强制将 mask 设为全 1
        if mask is not None:
            mask = mask.repeat(y.shape[0] // mask.shape[0], 1) if mask.shape[0] != y.shape[0] else mask
            mask = mask.squeeze(1).squeeze(1)
            mask = torch.where(force_drop_ids[:, None], 1, mask)
            if _xformers_available:
                y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = mask
        elif _xformers_available:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            raise ValueError(f"Attention type is not available due to _xformers_available={_xformers_available}.")


        # Transformer blocks
        for block_id, block in enumerate(self.blocks):
            x = auto_grad_checkpoint(
                block, x, y, t0, r0, role, y_lens, (self.h, self.w),
                **kwargs
            )  # (N, 1+K, T, D), support grad checkpoint
            
            if (block_id + 1) == 24:
            # if (block_id + 1) == 16:
                # print(f"x shape: {x.shape}")
                
                role_mask = (role == 0)  # bool tensor (N, 1+K)
                role0_feats = x[role_mask] # (num_role0, T, D)
                B_total, T, D = role0_feats.shape
                role0_feats_flat = role0_feats.reshape(B_total * T, D) # (B_total * T, D)
                # print(f"role_mask shape: {role_mask.shape}")
                # print(f"role0_feats shape: {role0_feats.shape}")
                
                projected_feats = self.projector(role0_feats_flat)  # (B_total * T, Z)
                Z = projected_feats.shape[-1]
                projected_feats = projected_feats.reshape(B_total, T, Z)  # (B_total, T, Z)
                zs = projected_feats

        # Final layer
        x = torch.unbind(x, dim=1)  ## 每张图拿出来，变为 list of [B, T, D]
        # 对每张图调用 T2IFinalLayer，将其从 patch-level 输出转为 latent 空间 [B, patch_size²*C, H//p, W//p]
        x = [final_layer(x[i], t + r[:, i, :]) for i, final_layer in enumerate(self.final_layers)]

        # Unpatchify
        x = [self.unpatchify(_x) for _x in x]  # tuple of (N, out_channels, H, W)
        x = torch.stack(x, dim=1)  # (N, 1+K, out_channels, H, W)

        # Detach role==1 and role==2 to prevent gradient flow
        # role==1 是条件输入，不能参与反向传播 role==2 是空图，更不能 backprop
        # 只有role==0 要生成的图才参与反向传播
        x = torch.where(torch.eq(role, 1)[..., None, None, None], x.detach(), x)
        x = torch.where(torch.eq(role, 2)[..., None, None, None], x.detach(), x)

        # return x
        return x, zs

    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        return self.forward(*args, **kwargs)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedders[0].patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    def initialize(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed:
        for x_embedder in self.x_embedders:
            w = x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Initialize role embedding:
        nn.init.normal_(self.role_embedder.weight, std=0.02)
        nn.init.normal_(self.role_block[1].weight, std=0.02)

        # Initialize domain embedding:
        nn.init.normal_(self.domain_embedding, std=0.02)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


@MODELS.register_module()
def P1_D28(**kwargs):
    return UMDM(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


@MODELS.register_module()
def P2_D28(**kwargs):
    return UMDM(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


@MODELS.register_module()
def P4_D28(**kwargs):
    return UMDM(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


@MODELS.register_module()
def P1_D20(**kwargs):
    # 20 layers, 1648.48M
    return UMDM(depth=20, hidden_size=2240, patch_size=1, num_heads=20, **kwargs)
    

@MODELS.register_module()
def _P2_D20(**kwargs):
    # 28 layers, 1648.48M
    return UMDM(depth=20, hidden_size=2240, patch_size=2, num_heads=20, **kwargs)


def build_mlp(hidden_size):
    return nn.Sequential(
        nn.Linear(hidden_size, 1024),
        nn.SiLU(),
        nn.Linear(1024, 1024),
        nn.SiLU(),
        nn.Linear(1024, 384),
    )
