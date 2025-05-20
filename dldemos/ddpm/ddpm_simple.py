import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
def img_plot(x) :                                                                   # 可视化功能
    images = x.cpu().numpy()
    images = (images - images.min()) / (images.max() - images.min())  # 归一化到 [0, 1] 范围

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i, 0], cmap='gray')  # 显示每张图片，灰度图
        ax.axis('off')
    plt.show()
class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
    # 正向    训练过程
    def sample_forward(self, x, t, eps=None):     # 生成t步加噪后的图像            # # x 512张图片       t 随机时间步512个         eps 噪声  512个
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)                     #  torch.Size([512])  tensor([524, 369,   3, 785, 259, 517, 157, 815,
        # alpha_bar.shape      torch.Size([512, 1, 1, 1])        [[[5.9952e-01]]],
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x       # torch.sqrt  求根 alpha_bar 从0.999开始，逐渐降低  是alpha第0-t步连乘
        return res                                                              # img_plot(x)    img_plot(res)
    # 反向 预测过程
    def sample_backward(self, img_shape, net, device, simple_var=True):         # sample_backward方法，该方法用于在反向过程中从噪声生成图像。
        x = torch.randn(img_shape).to(device)                                   # 生成一个形状为img_shape的随机噪声张量x

        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):                               # 使用sample_backward_step方法对图像进行反向采样，逐步减少噪声。
            x = self.sample_backward_step(x, t, net, simple_var)                # x 噪声图    img_plot(x)    一步一步去噪
            # 循环从时间步self.n_steps -1到0，逐步反向生成图像。在每个时间步t，调用sample_backward_step方法对当前图像x进行反向采样。


        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True):               # t     步数    单步去噪过程
        n = x_t.shape[0]

        # 创建一个张量t_tensor，形状为(n, 1)，其中每个元素都是当前的时间步t，并移动到与x_t相同的设备上。
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)                                                # 根据噪声图x_t推理噪声eps      t_tensor是步数



        if t == 0:
            noise = 0


        else:
            if simple_var:
                var = self.betas[t]                                             # 计算当前时间步的噪声方差var


            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
                

            noise = torch.randn_like(x_t)                                       # 生成与x_t形状相同的随机噪声noise。

            noise *= torch.sqrt(var)                                            # 将噪声乘以方差的平方根进行缩放。

        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])                               
        # 计算当前时间步的均值mean，基于当前图像张量x_t、预测噪声eps、self.alphas[t]和self.alpha_bars[t]。
        x_t = mean + noise

        return x_t
