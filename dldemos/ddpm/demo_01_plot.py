import matplotlib.pyplot as plt
import numpy as np
import torch

# # 示例张量 x_t，假设它是你要可视化的图像张量
# x_t = torch.randn(5, 1, 28, 28)  # 使用随机噪声作为示例，实际情况应替换为你的数据

# # 将张量转换为 numpy 数组并进行简单的预处理
# images = x_t.cpu().numpy()
# images = (images - images.min()) / (images.max() - images.min())  # 归一化到 [0, 1] 范围

# # 可视化
# fig, axes = plt.subplots(1, 5, figsize=(15, 3))
# for i, ax in enumerate(axes):
#     ax.imshow(images[i, 0], cmap='gray')  # 显示每张图片，灰度图
#     ax.axis('off')
# plt.show()


def img_plot(x) :                                                     # 可视化功能
    images = x.cpu().numpy()
    images = (images - images.min()) / (images.max() - images.min())  # 归一化到 [0, 1] 范围

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i, 0], cmap='gray')  # 显示每张图片，灰度图
        ax.axis('off')
    plt.show()

# if 
if __name__ == '__main__':
    # 示例张量 x_t，假设它是你要可视化的图像张量
    x_t = torch.randn(5, 1, 28, 28)  # 使用随机噪声作为示例，实际情况应替换为你的数据    
    img_plot(x_t)