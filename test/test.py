import torch
import matplotlib.pyplot as plt

def show_images(images: torch.Tensor):
    """Display a batch of images.

    Args:
        images: A tensor of shape (N, H, W, C) containing the images.
    """
    # 确保图像数据类型为 uint8
    if images.dtype != torch.uint8:
        images = images.to(torch.uint8)
    
    # 调整维度从 (N, H, W, C) 到 (N, C, H, W)
    images = images.permute(0, 3, 1, 2)
    
    # 展示每张图像
    n_images = images.shape[0]
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 2, 2))
    if n_images == 1:
        axes = [axes]
    for ax, img in zip(axes, images/255):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.show()

import torch
from torchvision.utils import save_image

def save_images(images: torch.Tensor, file_path: str, flag: str):
    """Save a batch of images to a file.

    Args:
        images: A tensor of shape (N, H, W, C) containing the images.
        file_path: The path to save the images to.
    """
    # 确保图像数据类型为 uint8
    if images.dtype != torch.uint8:
        images = images.to(torch.uint8)
    
    # 调整维度从 (N, H, W, C) 到 (N, C, H, W)
    images = images.permute(0, 3, 1, 2)
    
    # 保存图像
    if flag == "depth":
        images = (images - images.min()) / (images.max() - images.min())
        save_image(images, file_path, nrow=round(images.shape[0] ** 0.5))

    if flag == "rgb":
        save_image(images/255, file_path, nrow=round(images.shape[0] ** 0.5))


# 示例调用
# images = torch.randint(0, 256, (5, 480, 640, 3), dtype=torch.uint8)  # 假设有 5 张 480x640 的 RGB 图像
images_rgb = torch.load('images_tensor_rgb.pt')
images_depth = torch.load('images_tensor_depth.pt')
# show_images(images)
save_images(images_rgb, 'camera_rgb.png', 'rgb')
save_images(images_depth, 'camera_depth.png', 'depth')

from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights='/home/sangfor/Documents/IsaacLab/pretrain_model/resnet18-f37072fd.pth')
print('hello')