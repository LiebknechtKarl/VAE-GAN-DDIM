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
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from dldemos.ddpm.dataset import get_dataloader, get_img_shape
from dldemos.ddpm.ddpm_simple import DDPM
from dldemos.ddpm.network import (build_network, convnet_big_cfg,
                                  convnet_medium_cfg, convnet_small_cfg,
                                  unet_1_cfg, unet_res_cfg)

batch_size = 512
# n_epochs = 100
n_epochs = 1


def img_plot(x) :                                                                   # 可视化功能
    images = x.cpu().numpy()
    images = (images - images.min()) / (images.max() - images.min())  # 归一化到 [0, 1] 范围

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(images[i, 0], cmap='gray')  # 显示每张图片，灰度图
        ax.axis('off')
    plt.show()

def train(ddpm: DDPM, net, device='cuda', ckpt_path='dldemos/ddpm/model.pth'):      # 训练过程
    print('batch size:', batch_size)                                                # 512  batch_size
    n_steps = ddpm.n_steps                                                          # 1000
    dataloader = get_dataloader(batch_size)                                         # 数据加载器（dataloader）。
    net = net.to(device)
                        # UNet(
                        #     (pe): PositionalEncoding(
                        #         (embedding): Embedding(1000, 128)
                        #     )
                        #     (encoders): ModuleList(
                        #         (0): Sequential(
                        #         (0): UnetBlock(
                        #             (ln): LayerNorm((1, 28, 28), eps=1e-05, elementwise_affine=True)   ......

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    for e in range(n_epochs):                                                       # 设定的训练总轮数
        total_loss = 0

        for x, _ in dataloader:                                                     # x是手写数字图

            current_batch_size = x.shape[0]                                         # x   torch.Size([512, 1, 28, 28])  img_plot(x[0:5])   数字图

            x = x.to(device)                                                        # img_plot(x)

            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)        # 生成一个张量t，包含current_batch_size个在[0, n_steps)范围随机整数。这些整数表示时间步t，每个图像都会对应一个随机的时间步。
            # 整数表示时间步t，每个图像都会对应一个随机的时间步。                            # t  torch.Size([512])    tensor([411, 813, 774,  38, 285, 759, 642, 488, 303, 815, 855, 702, 650, 154,     ...
            
            eps = torch.randn_like(x).to(device)                                    # 噪声   torch.Size([512, 1, 28, 28])    img_plot(eps)

            # def sample_forward(self, x, t, eps=None):                             # # x 512张图片       t 随机时间步512个         eps 噪声  512个
            #     alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
            #     if eps is None:
            #         eps = torch.randn_like(x)
            #     res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
            #     return res
            x_t = ddpm.sample_forward(x, t, eps)                                    # x 512张图片       t 随机时间步512个         eps 噪声  512个
            # 调用DDPM类的sample_forward方法，对输入图像x进行前向采样，生成带噪声的图像x_t。t表示时间步，eps是随机噪声。



            eps_theta = net(x_t, t.reshape(current_batch_size, 1))                  # 预测噪声    将带噪声的图像x_t和时间步t（将t的形状调整为[current_batch_size, 1]）
                                                                                    # 作为输入，传入网络net，得到网络的输出eps_theta。这个输出是网络对噪声eps的预测。
            # net
            #     UNet(
            #     (pe): PositionalEncoding(
            #         (embedding): Embedding(1000, 128)
            #     )
            #     (encoders): ModuleList( ... ...            
            
            loss = loss_fn(eps_theta, eps)                                          # 计算预测噪声eps_theta与实际噪声eps之间的均方误差（MSE）损失。

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample_imgs(ddpm,
                net,
                output_path,
                n_sample=81,
                device='cuda',
                simple_var=True):
    

    net = net.to(device)                                                           # 训练的net


    net = net.eval()


    with torch.no_grad():


        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28                       # n_sample   81    
        # shape    (81, 1, 28, 28)

        imgs = ddpm.sample_backward(shape,                                         # shape    (81, 1, 28, 28)
                                    net,
                                    device=device,
                                    simple_var=simple_var).detach().cpu()
        

        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)


        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))



        imgs = imgs.numpy().astype(np.uint8)



        cv2.imwrite(output_path, imgs)


configs = [
    convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg,
    unet_res_cfg
]

if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    n_steps = 1000
    config_id = 4
    device = 'cuda'
    model_path = 'dldemos/ddpm/model_unet_res.pth'

    config = configs[config_id]
                                                                            # n_steps     1000
    net = build_network(config, n_steps)                                    # config   {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128, 'residual': True}
    # net
    #     UNet(
    #     (pe): PositionalEncoding(
    #         (embedding): Embedding(1000, 128)
    #     )
    #     (encoders): ModuleList( ... ...
    ddpm = DDPM(device, n_steps)                                            # 设定总加噪时间步
    # <dldemos.ddpm.ddpm_simple.DDPM object at 0x7f8dd8e1b160>

    train(ddpm, net, device=device, ckpt_path=model_path)                   # 训练噪声预测

    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, 'work_dirs/diffusion.jpg', device=device)        # 预测  根据训练得到的权重
