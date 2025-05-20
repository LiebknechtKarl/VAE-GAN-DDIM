import torch
from tqdm import tqdm

from dldemos.ddim.ddpm import DDPM
# 使用了扩散模型的逆过程，通过逐步的反向采样生成最终图像。
# 核心的步骤包括计算时间步、预测去噪噪声、以及基于噪声和模型输出更新样本。
class DDIM(DDPM):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_backward(self,
                        img_or_shape,
                        net,
                        device,
                        simple_var=True,
                        ddim_step=20,
                        eta=1):
    # def sample_backward(self, img_or_shape,  net, device, simple_var=True,  ddim_step=20, eta=1):        


        if simple_var:
            eta = 1
        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        # tensor([1000,  999,  998,  ...,    2,    1,    0], device='cuda:0')


        if isinstance(img_or_shape, torch.Tensor):  # # img_or_shape   (64, 1, 28, 28)   检查img_or_shape是否为张量
            x = img_or_shape
        else:
            x = torch.randn(img_or_shape).to(device)
        #         # x 变为随机噪声
        # x.max()   tensor(5.1427, device='cuda:0')      x.min()  tensor(-3.9412, device='cuda:0')


        batch_size = x.shape[0]         # 64




        net = net.to(device)
        for i in tqdm(range(1, ddim_step + 1),
                      f'DDIM sampling with eta {eta} simple_var {simple_var}'):
            # 1,2,3,4, ...... 999,1000 

            # i = 1
            cur_t = ts[i - 1] - 1               # tensor(999, device='cuda:0')
            prev_t = ts[i] - 1
            # ts        tensor([1000,  999,  998,  ...,    2,    1,    0], device='cuda:0')
            # cur_t      tensor(999, device='cuda:0')
            # prev_t     tensor(998, device='cuda:0')
            ab_cur = self.alpha_bars[cur_t]  # self.alpha_bars   tensor([9.9990e-01, 9.9978e-01, 9.9964e-01, 9.9948e-01, 9.9930e-01, 9.9910e-01,
            # ... ...   4.5545e-05, 4.4639e-05, 4.3750e-05, 4.2877e-05, 4.2022e-05, 4.1182e-05, 4.0358e-05], device='cuda:0')
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1
            # ab_cur      tensor(4.0358e-05, device='cuda:0')
            # ab_prev     tensor(4.1182e-05, device='cuda:0')

            t_tensor = torch.tensor([cur_t] * batch_size,
                                    dtype=torch.long).to(device).unsqueeze(1)
            # cur_t   tensor(999, device='cuda:0')           batch_size  64
            # t_tensor       tensor([[999],[999],[999],[999],[999],    ....        [999]], device='cuda:0')


            eps = net(x, t_tensor)  # x  带噪声图片        t_tensor  tensor([[999],[999],[999],[999],[999],    ....        [999]], device='cuda:0')      
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            #     eta  0               ab_prev  tensor(4.1182e-05, device='cuda:0') 
            #     ab_cur   tensor(4.0358e-05, device='cuda:0')          ab_prev  tensor(4.1182e-05, device='cuda:0')
            #     var   tensor(0., device='cuda:0')  =  0*(1-4.1182e-05)/(1-4.0358e-05)*(1-4.0358e-05/4.1182e-05)

            noise = torch.randn_like(x)
            # x    x.shape     torch.Size([64, 1, 28, 28])        tensor([[[[-1.8415e-01,  5.0570e-01, -6.0770e-01,  ..., -2.0750e-01,-3.5052e-02,  1.7301e+00],[ 8.8746e-01,  1.3485e-01,  3.3783e-01,  ...,  3.8983e-01,  ... ...
            # noise   tensor([[[[ 2.9141e-01, -2.6334e-01,  1.4561e-01,  ...,  1.2378e+00,3.6450e-01,  1.0059e+00],[-1.1165e+00, -1.1135e+00,  8.7409e-01,  ..., -4.1837e-01,



            first_term = (ab_prev / ab_cur)**0.5 * x
            #



            second_term = ((1 - ab_prev - var)**0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * eps
            



            if simple_var:
                third_term = (1 - ab_cur / ab_prev)**0.5 * noise




            else:
                third_term = var**0.5 * noise



                
            x = first_term + second_term + third_term

        return x
