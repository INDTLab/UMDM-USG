import os
from scipy.io import loadmat

import torch

from .utils import check_tensor


class SegADE20KPostProcessor:

    colors150 = loadmat(os.path.join(os.path.dirname(__file__), 'color150.mat'))['colors']
    colors150 = torch.from_numpy(colors150).float()

    colors12 = torch.tensor([
        [0, 0, 0],  
        [0, 0, 255],    
        [0, 255, 0],    
        [0, 255, 255],  
        [255, 0, 0],    
        [255, 0, 255],  
        [255, 255, 0], 
        [255, 255, 255],
    ], dtype=torch.float)

    def __init__(self, color_scheme: str = 'colors150', only_return_image: bool = False):
        if color_scheme not in ['colors150', 'colors12']:
            raise ValueError(f"Unknown color scheme: {color_scheme}")
        self.color_scheme = color_scheme
        self.only_return_image = only_return_image

    def __call__(self, image: torch.Tensor):
        """Quantize the input image to the nearest color in the color palette."""
        check_tensor(image, ndim=4, min_value=-1, max_value=1)
        B, C, H, W = image.shape
        if self.color_scheme == 'colors150':
            colors = self.colors150.to(image.device)
            colors = torch.cat([torch.zeros_like(colors[:1]), colors], dim=0)
        else:
            colors = self.colors12.to(image.device)
        image = (image.float() + 1.) / 2.
        image = (image * 255).reshape(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        dist = torch.sum((image[:, :, None] - colors[None, None]) ** 2, dim=-1)  # (B, H*W, classes)
        index = torch.argmin(dist, dim=-1).reshape(B, H, W)
        image = colors[index].reshape(B, H, W, C).permute(0, 3, 1, 2) / 255.  # (B, 3, H, W)
        if self.only_return_image:
            return image
        return {'image': image, 'index': index}

    def index2color(self, index: torch.Tensor):
        if self.color_scheme == 'colors150':
            colors, num_colors = self.colors150.to(index.device), 151
            colors = torch.cat([torch.zeros_like(colors[:1]), colors], dim=0)
        else:
            colors, num_colors = self.colors12.to(index.device), 12
        assert (0 <= index).all() and (index < num_colors).all()
        index = index.long()
        image = colors[index].reshape(*index.shape, 3) / 255.
        if image.ndim == 4:
            image = image.permute(0, 3, 1, 2)  # (B, C, H, W)
        elif image.ndim == 3:
            image = image.permute(2, 0, 1)  # (C, H, W)
        else:
            raise ValueError("Index tensor must be 3D (H, W) or 4D (B, H, W)")
        return image
