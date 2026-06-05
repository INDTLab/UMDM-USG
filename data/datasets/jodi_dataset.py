import os
import re
import json
import numpy as np
from PIL import Image
from random import shuffle

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from data.builder import DATASETS

Image.MAX_IMAGE_PIXELS = None


def get_combinition(n: int):
    return (np.arange(2**n, dtype=np.long)[:, None] >> np.arange(n-1, -1, -1)) & 1


def clean_filename(filename):
    match = re.match(r"^(.*?)(\.jpg|\.png)(?:\.(jpg|png))*$", filename, re.IGNORECASE)
    return match.group(1) + match.group(2) if match else filename


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


# image.jsonl: 每行是一个 JSON 字典
# 每张图片对应两份描述文件 .caption.json: 多个 caption .info.json: 包含原图 height, width
# annotation_...: 是每个条件图像的文件夹+索引（比如 annotation_seg 存 mask）
@DATASETS.register_module()
class JodiDataset(Dataset):

    aspect_ratio = {
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

    ratio_nums = {k: 0 for k in aspect_ratio.keys()}

    def __init__(
            self,
            data_dir: str,
            conditions: list[str],                          # 条件名列表，如 ["seg", "depth", "openpose"]
            split: str = None,
            tasks: list[str] = ("j", "c", "p"),             # 任务模式：j (joint), c (conditional), p (perceptual)
            # caption_model_probs: dict[str, float] = None,   # 多 caption 模型选择的概率
            repeat_time: int = 1,                           # 重复样本倍数
            use_empty_openpose_image: bool = True,          # 是否填充黑图
    ):
        self.data_dir = os.path.expanduser(data_dir)
        self.conditions = conditions
        self.split = split
        self.tasks = tasks
        # self.caption_model_probs = caption_model_probs
        self.repeat_time = repeat_time
        self.use_empty_openpose_image = use_empty_openpose_image

        self.data = self.load_data()
        self.data = self.data * repeat_time

        # 这两个是 2^K 维度的组合编码，用于动态构建 role 标签
        self.role_cond_gen = get_combinition(len(self.conditions)) + 1  # (2^K, K)
        self.role_img_perc = get_combinition(len(self.conditions)) * 2  # (2^K, K)

    # 加载 image.jsonl 和每个 annotation_*.jsonl，生成完整的数据样本列表
    # 每个样本字典 info 中包含：
    # {
    #   "image_path": ...,
    #   "caption": ...,
    #   "info": ...,
    #   "annotation_seg": ...,
    #   "annotation_depth": ...,
    # } 类似这样  
    
    def load_data(self):
        data = []
        jsonl_path = os.path.join(self.data_dir, "output.jsonl")
        assert os.path.isfile(jsonl_path), f"JSONL file {jsonl_path} does not exist."
    
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
    
                image_rel_path = entry["image"]
                image_path = os.path.join(self.data_dir, image_rel_path)
                image_id = clean_filename(os.path.basename(image_rel_path))
    
                assert os.path.isfile(image_path), f"Image file {image_path} not found."
    

                for key, val in entry.items():
                    if key.startswith("annotation_"):
                        # print(f"{key} : '{val}'") 
                        ann_path = os.path.join(self.data_dir, val)
                        # print(f"Checking file: {ann_path}")
                        assert os.path.isfile(ann_path), f"Annotation file {ann_path} not found."
    

                item = {
                    "image_path": image_path,
                    "caption": entry["caption"],
                    "info": entry["info"],
                }

                for key, val in entry.items():
                    if key.startswith("annotation_"):
                        item[key] = os.path.join(self.data_dir, val)
    
                # print(f"Loaded item for image_id={image_id}: keys = {list(item.keys())}")
                data.append((image_id, item))
    
        return data

        
    def __len__(self):
        return len(self.data)

    # 给定 index，返回一个样本的字典
    # {
    #   "img": Tensor[K+1, C, H, W],
    #   "text": prompt string,
    #   "role": Tensor[K+1]  # 0=生成的，1=条件图，2=空图
    # }
    
    def __getitem__(self, index: int):
        image_id, info = self.data[index]
        image_path = info["image_path"]
    
        image_info = info["info"]  
        transform = self.get_transform(image_info["height"], image_info["width"])
    
        img = [transform(Image.open(image_path).convert("RGB"))]
    
        role, cnt = [], 0
        for condition in self.conditions:
        
            ann_path = info.get(f"annotation_{condition}", None)
            
            # print(f"[DEBUG] condition: {condition}, ann_path: {ann_path}")
            
            if ann_path is None:
                if self.use_empty_openpose_image and "openpose" in condition:
                    h, w = img[0].shape[-2:] if isinstance(img[0], torch.Tensor) else (img[0].height, img[0].width)
                    img.append(transform(self.get_empty_openpose_image(h, w)))
                    role.append(None)
                    cnt += 1
                else:
                    img.append(torch.zeros_like(img[0]) if isinstance(img[0], torch.Tensor) else None)
                    role.append(2)
            else:
                img.append(transform(Image.open(ann_path).convert("RGB")))
                role.append(None)
                cnt += 1
                
        # print(f"Final role: {role}, cnt={cnt}")
    
        if isinstance(img[0], torch.Tensor):
            img = torch.stack(img, dim=0)
            
            
    
        text = info["caption"]

        # 这里！！！！重要重要！！！！
        # role的分配
        # print("*******************************************")
        
        task = self.tasks[np.random.randint(len(self.tasks))]
        # print(f"[DEBUG] Selected task: {task}")
        
        
        # cnt = len([r for r in role if r is not None])
        # print(f"[DEBUG] Number of available conditions (cnt): {cnt}")
        # print(f"[DEBUG] Original role list: {role}")
        
        
        # 文生图 文本生成所有 全部0
        if task == "j":  # joint generation
            role = [0] + [r if r is not None else 0 for r in role]
            # print(f"[DEBUG] Joint generation role: {role}")
            
            
            
        # 条件生成 condition图 → 生成image  随机选择哪些作为条件  随机几个为1，其他的为0
        elif task == "c":  # condition generation
            _role = self.role_cond_gen[:2**cnt, len(self.conditions)-cnt:]
            _role = _role[np.random.randint(len(_role))].tolist()
            role = [0] + [r if r is not None else _role.pop() for r in role]
            # print(f"[DEBUG] Final role after conditional generation: {role}")
            
            
        # 感知生成  image设为输入 → 预测condition            随机几个为0，其他的为1
        elif task == "p":  # image perception
            _role = self.role_img_perc[:2**cnt, len(self.conditions)-cnt:][:-1]
            
            _role = _role[np.random.randint(len(_role))].tolist()
            
            role = [1] + [r if r is not None else _role.pop() for r in role]
            # print(f"[DEBUG] Final role after perception generation: {role}")
        
        
        else:
            raise ValueError(f"Unknown task {task}")
        role = torch.tensor(role).long()

        return dict(img=img, text=text, role=role)

    # 根据原图尺寸找到最接近的 aspect_ratio，裁剪为固定尺寸
    def get_transform(self, height: int, width: int):
        closest_size, closest_ratio = get_closest_ratio(height, width, self.aspect_ratio)
        closest_size = list(map(lambda x: int(x), closest_size))

        if closest_size[0] / height > closest_size[1] / width:
            resize_size = closest_size[0], int(width * closest_size[0] / height)
        else:
            resize_size = int(height * closest_size[1] / width), closest_size[1]

        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(closest_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def get_data_info(self, index: int):
        image_id, info = self.data[index]
        image_info = info["info"]  # 直接使用已经加载好的字典
        return {"height": image_info["height"], "width": image_info["width"]}

    @staticmethod
    def get_empty_openpose_image(h, w):  # pure black image
        return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))

# 如果你有多个数据集（比如多个任务域或子数据集），这个类将它们拼接在一起，并打乱样本顺序
class RandomConcatJodiDataset(Dataset):
    def __init__(self, datasets: list[JodiDataset]):
        self.datasets = datasets
        self.indices = []
        for k, dataset in enumerate(self.datasets):
            self.indices.extend(list(zip([k] * len(dataset), range(len(dataset)))))
        shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        dataset_idx, sample_idx = self.indices[index]
        dataset = self.datasets[dataset_idx]
        return dataset[sample_idx]

    def get_data_info(self, index: int):
        dataset_idx, sample_idx = self.indices[index]
        dataset = self.datasets[dataset_idx]
        return dataset.get_data_info(sample_idx)

    @property
    def aspect_ratio(self):
        return self.datasets[0].aspect_ratio

    @property
    def ratio_nums(self):
        return self.datasets[0].ratio_nums
