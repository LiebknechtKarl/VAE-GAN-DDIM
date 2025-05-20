import torch
from tqdm import tqdm


class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        # betas     torch.Size([1000])           tensor([1.0000e-04, 1.1992e-04, 1.3984e-04, 1.5976e-04, 1.7968e-04, 1.9960e-04,        

        alphas = 1 - betas
        # alphas      torch.Size([1000])        tensor([0.9999, 0.9999, 0.9999, 0.9998, 0.9998, 0.9998, ... ...   0.9800, 0.9800,0.9800], device='cuda:0')        

        alpha_bars = torch.empty_like(alphas)
        # torch.empty()就是创建一个使用未初始化值填满的tensor，至于      tensor([-3.6893e+19,  6.0240e-01,  1.3984e-04,  1.5976e-04,  1.7968e-04,

        product = 1


        for i, alpha in enumerate(alphas):      #  tensor([0.9999, 0.9999, 0.9999, 0.9998, 0.9998, 0.9998, ... ...   0.9800, 0.9800,0.9800], device='cuda:0')        
            product *= alpha
            alpha_bars[i] = product             # 连乘 alphas 内部元素
        # alpha_bars      torch.Size([1000])       tensor([9.9990e-01, 9.9978e-01, 9.9964e-01, 9.9948e-01, 9.9930e-01, 9.9910e-01, 。。。。 4.2877e-05, 4.2022e-05, 4.1182e-05, 4.0358e-05], device='cuda:0')

        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars


    # sample_forward 方法用于在前向过程中对图像进行采样。
    # 参数 x 是输入图像，t 是时间步，eps 是噪声（如果未提供则生成随机噪声）。
    # 根据 alpha_bar 计算结果并返回。      返回加噪结果
    def sample_forward(self, x, t, eps=None):
        # x           # torch.Size([512, 1, 28, 28])                tensor([[[[-1., -1., -1.,  ..., -1., -1., -
        # t           # torch.Size([512])        tensor([505, 625, 989,   7, 919, 786,  18, 407, 269, 678, 602, 384, 935,  42,
        # eps         # eps.shape     torch.Size([512, 1, 28, 28])      tensor([[[[-1.8540, -1.3416,  0.7494,  ..., -0.6358,  2.6665, ...

        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        # alpha_bar.shape           torch.Size([512, 1, 1, 1])      tensor([[[[3.0795e-01]]], [[[2.8117e-01]]],[[[3.0584e-01]]], ... ...

        if eps is None:                             # 
            eps = torch.randn_like(x)

        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x           #          # 噪声公式

        return res
    # sample_backward 函数用于从扩散模型的噪声中逐步生成图像。以下是对代码的详细分析：
    def sample_backward(self, img_or_shape, net, device, simple_var=True):
        # img_or_shape          (64, 1, 28, 28)



        if isinstance(img_or_shape, torch.Tensor): # img_or_shape      (64, 1, 28, 28)      检查img_or_shape是否为张量
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)
        # x 变为随机噪声
        # x.max()   tensor(5.1427, device='cuda:0')      x.min()  tensor(-3.9412, device='cuda:0')


        net = net.to(device)




        for t in tqdm(range(self.n_steps - 1, -1, -1), 'DDPM sampling'):   # tqdm库来显示进度条
                        #  (1000         -1 , -1,-1)     t    999,998,997,996,995 ... ...



            x = self.sample_backward_step(x, t, net, simple_var)





        return x            # torch.Size([64, 1, 28, 28])

    def sample_backward_step(self, x_t, t, net, simple_var=True):
        #  x_t.shape  torch.Size([64, 1, 28, 28])              t  999       
        n = x_t.shape[0]                # 其实就是bs    64  
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)   # .shape  torch.Size([64, 1])  bs个t张量
        eps = net(x_t, t_tensor)            # unet  计算噪声        
        # eps.shape             torch.Size([64, 1, 28, 28])
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

        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])           # 去噪公式
        x_t = mean + noise

        return x_t
