import torch
import matplotlib.pyplot as plt
import numpy as np
import torch

# 代码其实就是逐步加噪

def img_plot(x) :                                                           # 这是我自己写的看图的函数
    images = x.cpu().numpy()
    images = (images - images.min()) / (images.max() - images.min())        # 归一化到 [0, 1] 范围

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i, 0], cmap='gray')                                # 显示每张图片，灰度图
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
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)


    # sample_forward 方法用于在前向过程中对图像进行采样。
    # 参数 x 是输入图像，t 是时间步，eps 是噪声（如果未提供则生成随机噪声）。
    # 根据 alpha_bar 计算结果并返回。
    def sample_forward(self, x, t, eps=None):                                            #  x 是输入图像，t 是时间步    11
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)                              # 从 self.alpha_bars 中提取时间步 t 对应的值，并将其形状调整为 (-1, 1, 1, 1)，使其适应输入图像 x 的形状。
        # 假设 self.alpha_bars 是一个形状为 (100,) 的一维张量，表示 100 个时间步的 alpha_bar 值。当前时间步 t 是 50，那么 self.alpha_bars[t] 提取的是第 50 个时间步的 alpha_bar 值。
        # tensor([0.9999, 0.9996, 0.9991, 0.9984, 0.9975, 0.9964, 0.9951, 0.9936, 0.9919,
        # 0.9900, 0.9879, 0.9856, 0.9832, 0.9805, 0.9776, 0.9746, 0.9713, 0.9679,0.9643, 0.9606, 0.9566, 0.9525, 0.9482, 0.9437, 0.9390, 0.9342, 0.9292,
        # 0.9241, 0.9188, 0.9134, 0.9078, 0.9020, 0.8961, 0.8901, 0.8839, 0.8776,0.8712, 0.8646, 0.8579, 0.8511, 0.8442, 0.8371, 0.8300, 0.8227, 0.8154,
        # 0.8079, 0.8004, 0.7927, 0.7850, 0.7772, 0.7693, 0.7613, 0.7533, 0.7452,0.7370, 0.7288, 0.7205, 0.7122, 0.7038, 0.6954, 0.6870, 0.6785, 0.6699,
        # 0.6614, 0.6528, 0.6442, 0.6356, 0.6270, 0.6184, 0.6097, 0.6011, 0.5924,0.5838, 0.5752, 0.5666, 0.5580, 0.5494, 0.5408, 0.5323, 0.5238, 0.5153,
        # 0.5069, 0.4985, 0.4901, 0.4818, 0.4735, 0.4653, 0.4571, 0.4489, 0.4409,0.4329, 0.4249, 0.4170, 0.4092, 0.4014, 0.3937, 0.3860, 0.3785, 0.3710,0.3636], device='cuda:0')
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x               # 噪声公式
        return res

    def sample_backward(self,
                        img_shape,
                        net,
                        device,
                        simple_var=True,
                        clip_x0=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):

        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                   eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t


def visualize_forward():
    import cv2
    import einops
    import numpy as np

    from dldemos.ddpm.dataset import get_dataloader         # 数据加载器（dataloader）

    n_steps = 100
    device = 'cuda'
    dataloader = get_dataloader(5)                          # 参数 5 表示批量大小（batch size）
    x, _ = next(iter(dataloader))                           # 获取数据加载器中下一批数据。返回包含数据和标签的元组 (data, labels)，但只取数据 x，忽略标签 _。
    x = x.to(device)                                        # x.shape       torch.Size([5, 1, 28, 28])

    ddpm = DDPM(device, n_steps)                            # 创建 DDPM 类的实例。device 是计算设备，n_steps 是扩散过程的步数。
    xts = []                                                # 初始化一个空列表 xts，用于存储在不同时间步的采样结果。
    percents = torch.linspace(0, 0.99, 10)                  # 0 到 0.99 的等间隔的 10 个数值的张量 percents。这些值表示扩散过程中的不同时间点。 tensor([0.0000, 0.1100, 0.2200, 0.3300, 0.4400, 0.5500, 0.6600, 0.7700, 0.8800,0.9900])
    for percent in percents:                                # tensor([0.0000, 0.1100, 0.2200, 0.3300, 0.4400, 0.5500, 0.6600, 0.7700, 0.8800,0.9900])
        t = torch.tensor([int(n_steps * percent)])          # 将百分比转换为实际时间步 t。 0   11   22    33   44    55    ......
        t = t.unsqueeze(1)                                  # 将 t 扩展一个维度，使其形状从 (1,) 变为 (1, 1)，以便与 sample_forward 函数的输入匹配。
        x_t = ddpm.sample_forward(x, t)                     # torch.Size([5, 1, 28, 28])           tensor([[0]])
        # x_t           torch.Size([5, 1, 28, 28])
        # tensor([[[[-0.9894, -1.0050, -0.9946,  ..., -0.9959, -1.0002, -0.9986],
        #         [-1.0077, -1.0148, -1.0043,  ..., -1.0060, -0.9960, -0.9981],
        # ... ...
        xts.append(x_t)
        #       img_plot(x_t)



    res = torch.stack(xts, 0)                               # len(xts)  10        逐步加噪的过程
    res = einops.rearrange(res, 'n1 n2 c h w -> (n2 h) (n1 w) c')
    res = (res.clip(-1, 1) + 1) / 2 * 255
    res = res.cpu().numpy().astype(np.uint8)

    cv2.imwrite('work_dirs/diffusion_forward.jpg', res)


def main():
    visualize_forward()


if __name__ == '__main__':
    main()
