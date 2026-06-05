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

import os
import warnings
import pyrallis
from dataclasses import dataclass, field
from typing import Tuple, List
from PIL import Image

import torch
import torchvision.transforms as T

warnings.filterwarnings("ignore")  # ignore warning


from diffusion import DPMS
from model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from model.utils import get_weight_dtype, prepare_prompt_ar
from utils.config import BaseConfig, ModelConfig, AEConfig, TextEncoderConfig, SchedulerConfig, model_init_config
from utils.logger import get_root_logger

from tools.download import find_model

# 微调Sana实现的 这里是推理

# 将输入图像转换为 3D Tensor 并标准化为 [-1, 1] 区间
def read_image(image):
    if isinstance(image, str):
        assert os.path.exists(image), f"Image {image} does not exist."
        image = Image.open(image).convert("RGB")
        transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        image = transform(image)
    elif isinstance(image, Image.Image):
        transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        image = transform(image)
    elif isinstance(image, torch.Tensor):
        assert image.ndim == 3, "Image tensor should be 3D."
    else:
        raise TypeError("Unsupported image type. Expected str, PIL Image, or Tensor.")
    return image

# 将任意输入尺寸根据 aspect ratio 映射到最近的标准尺寸
# 根据给定高宽计算 aspect ratio（宽高比），将其匹配到最近的标准比例，并返回标准尺寸 我要用的应该得改一下对应的字典，我好多1.0的扩展到1024不太行，太大了
def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])


@dataclass
class JodiInference(BaseConfig):
    model: ModelConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig
    config: str = "./configs/inference.yaml"
    conditions: List[str] = field(default_factory=list)
    work_dir: str = "output/"


class JodiPipeline:
    def __init__(
        self,
        config: str,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        # 使用 pyrallis 加载配置
        config = pyrallis.load(JodiInference, open(config))
        self.config = config
        self.device = device
        self.logger = get_root_logger()
        self.progress_fn = lambda progress, desc: None

        # set some hyperparameters
        self.image_size = config.model.image_size
        self.latent_size = self.image_size // config.vae.vae_downsample_rate
        self.max_sequence_length = config.text_encoder.model_max_length
        self.flow_shift = config.scheduler.flow_shift

        self.weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)

        self.logger.info(f"flow_shift: {self.flow_shift}")
        self.logger.info(f"Inference with {self.weight_dtype}")

        self.num_conditions = len(config.conditions)

        # 1. build vae and text encoder
        self.vae = self.build_vae(config.vae)
        self.tokenizer, self.text_encoder = self.build_text_encoder(config.text_encoder)

        # 2. build Jodi
        self.model = self.build_jodi(config).to(self.device)

        # 3. pre-compute null embedding 提前缓存空 prompt 的编码结果
        with torch.no_grad():
            null_caption_token = self.tokenizer(
                "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(
                null_caption_token.input_ids, null_caption_token.attention_mask
            )[0]

    @property
    def base_ratios(self):
        return {
            "0.50": [512.0, 1024.0],  # count: 110   2:1
            "0.55": [768.0, 1408.0],  # count: 2690  11:6
            "0.60": [640.0, 1024.0],  # count: 280   8:5
            "0.65": [640.0, 960.0],  # count: 5936   3:2
            "0.70": [704.0, 960.0],  # count: 739    4:3
            "0.75": [576.0, 704.0],  # count: 14507  6:5
            "0.80": [768.0, 960.0],  # count: 301    5:4
            "0.85": [832.0, 960.0],  # count: 109    6:5
            "1.00": [384.0, 384.0],  # count: 9381   1:1
            "1.35": [960.0, 704.0],  # count: 341    3:4
            "1.50": [896.0, 640.0],  # count: 602    2:3
        }

    # 构建 VAE
    def build_vae(self, config):
        vae = get_vae(config.vae_type, config.vae_pretrained, self.device).to(self.vae_dtype)
        return vae

    # 加载分词器（Tokenizer）和文本编码器
    def build_text_encoder(self, config):
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder_name, device=self.device)
        return tokenizer, text_encoder

    # 构建jodi
    def build_jodi(self, config):
        # model setting
        model_kwargs = model_init_config(config, latent_size=self.latent_size)
        model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
            num_conditions=self.num_conditions,
            **model_kwargs,
        )
        self.logger.info(f"use_fp32_attention: {model.fp32_attention}")
        # 显示模型参数数量
        self.logger.info(
            f"{model.__class__.__name__}:{config.model.model},"
            f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        return model

    # 读取权重
    def from_pretrained(self, model_path):
        state_dict = find_model(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        # 如果含有 pos_embed，则手动删除
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.weight_dtype)

        self.logger.info(f"Generating sample from ckpt: {model_path}")
        self.logger.warning(f"Missing keys: {missing}")
        self.logger.warning(f"Unexpected keys: {unexpected}")

    def register_progress_bar(self, progress_fn=None):
        self.progress_fn = progress_fn if progress_fn is not None else self.progress_fn

    # 执行推理
    @torch.inference_mode()
    def __call__(
        self,
        images,
        role,                   # 每张图对应角色
        prompt="",
        height=1024,
        width=1024,
        negative_prompt="",     # 反向提示
        num_inference_steps=20,
        guidance_scale=4.5,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
    ):
        ori_height, ori_width = height, width
        # 处理分辨率
        height, width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        latent_size_h, latent_size_w = (
            height // self.config.vae.vae_downsample_rate,
            width // self.config.vae.vae_downsample_rate,
        )

        # pre-compute negative embedding  负向 prompt 编码：若有，编码 negative_prompt 为 null_caption_embs
        if negative_prompt != "":
            null_caption_token = self.tokenizer(
                negative_prompt,
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(
                null_caption_token.input_ids, null_caption_token.attention_mask
            )[0]

        # compute clean_x
        # 根据 role，对输入的条件图像进行处理并编码为 latent
        if len(images) != 1 + self.num_conditions:
            raise ValueError(f"Number of images {len(images)} != {1 + self.num_conditions}.")
        if len(role) != 1 + self.num_conditions:
            raise ValueError(f"Number of roles {len(role)} != {1 + self.num_conditions}.")

        # 初始化了一个列表（长度为 1 + 条件数），每个元素是一个全 0 的 latent 张量
        clean_x = [
            torch.zeros(
                1,
                self.config.vae.vae_latent_dim,
                latent_size_h,
                latent_size_w,
                device=self.device,
                dtype=self.vae_dtype,
            )
        ] * (self.num_conditions + 1)

        # 逐个处理 images 中 role[i] == 1（即：条件图像）
        for i, image in enumerate(images):
            if role[i] == 1:
                assert image is not None   # 图像预处理 → resize + crop → 用 vae_encode() 编码成 latent 向量
                image = read_image(image).unsqueeze(0).to(self.device, self.vae_dtype)

                image_height, image_width = image.shape[-2:]
                if height / image_height > width / image_width:
                    resize_size = height, int(image_width * height / image_height)
                else:
                    resize_size = int(image_height * width / image_width), width

                resize_and_crop = T.Compose([
                    T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                    T.CenterCrop((height, width)),
                ])
                image = resize_and_crop(image)
                # # (1, 1+K, 32, 32, 32) # 替换掉原来的空位（clean_x[i] = ...）
                clean_x[i] = vae_encode(
                    self.config.vae.vae_type, self.vae, image, self.config.vae.sample_posterior, self.device
                )
        clean_x = torch.stack(clean_x, dim=1)   # (1, 1+K, C, H, W)
        role = torch.tensor(role).unsqueeze(0)  # (1, 1+K)
        role = role.to(dtype=torch.long, device=self.device)

        # 构造 Prompt 列表 重复 prompt N 次（每张图都用同一个 prompt）
        # 使用 prepare_prompt_ar 处理，加上分辨率信息
        prompts = [
            prepare_prompt_ar(prompt, self.base_ratios, device=self.device, show=False)[0].strip()
            for _ in range(num_images_per_prompt)
        ]

        # 如果有提示文本，就把提示文本加进去
        with torch.no_grad():
            # prepare text feature
            if not self.config.text_encoder.chi_prompt:
                max_length_all = self.config.text_encoder.model_max_length
                prompts_all = prompts
            else:
                # 拼接在前：[chi_prompt] + [prompt]
                chi_prompt = "\n".join(self.config.text_encoder.chi_prompt)
                prompts_all = [chi_prompt + prompt for prompt in prompts]
                num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
                max_length_all = (
                    num_chi_prompt_tokens + self.config.text_encoder.model_max_length - 2
                )  # magic number 2: [bos], [_] 为了不超出长度，重新计算 max_length_all（加上前缀的 token 数）减去两个特殊 token 预留位（[bos], [_]）

            # 文本编码
            # tokenizer 编码 → encoder 输出 → 取出 select token → 形成 caption_embs
            caption_token = self.tokenizer(
                prompts_all,
                max_length=max_length_all,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device=self.device)

            select_index = [0] + list(range(-self.config.text_encoder.model_max_length + 1, 0))  # 选择第一个 token 和最后 N 个有效 token（排除前缀干扰）
            # caption_embs: [N, 1, T, D]
            caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
                :, :, select_index
            ].to(self.weight_dtype)
            emb_masks = caption_token.attention_mask[:, select_index]
            # null_y 是负向 prompt 的默认 latent 表达（可选 CFG 时使用）
            null_y = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None].to(self.weight_dtype)

            n = len(prompts)
            # 准备噪声
            if latents is None:
                z = torch.randn(
                    n,
                    1 + self.num_conditions,
                    self.config.vae.vae_latent_dim,
                    latent_size_h,
                    latent_size_w,
                    generator=generator,
                    device=self.device,
                )
            else:
                assert latents.shape == (
                    n,
                    1 + self.num_conditions,
                    self.config.vae.vae_latent_dim,
                    latent_size_h,
                    latent_size_w,
                )
                z = latents.to(self.device)

            # 扩展到 batch size = n，即每个样本都共享相同的 clean_x 和 role
            role = role.repeat(n, 1)
            clean_x = clean_x.repeat(n, 1, 1, 1, 1)

            model_kwargs = dict(mask=emb_masks, role=role, clean_x=clean_x)
            scheduler = DPMS(
                self.model,
                condition=caption_embs,
                uncondition=null_y,
                cfg_scale=guidance_scale,
                model_type="flow",
                model_kwargs=model_kwargs,
                schedule="FLOW",
            )

            scheduler.register_progress_bar(self.progress_fn)
            sample = scheduler.sample(
                z,
                steps=num_inference_steps,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=self.flow_shift,
            )

        # 对 role==1 的位置，将生成 latent 替换为原始 clean_x，确保条件图不被更改
        sample = torch.where(torch.eq(role, 1)[:, :, None, None, None], clean_x, sample)
        sample = sample.to(self.vae_dtype)
        sample = torch.unbind(sample, dim=1) # 分离每个通道
        # 输出的是多个 latent，分别对应目标图 + 条件图
        with torch.no_grad():
            sample = [vae_decode(self.config.vae.vae_type, self.vae, s) for s in sample]
        resize = T.Resize((ori_height, ori_width), interpolation=T.InterpolationMode.BILINEAR)
        sample = [resize(s).clamp(-1, 1) for s in sample]
        return sample
        # 返回的是一个列表，第一张是生成结果，后面是条件图
