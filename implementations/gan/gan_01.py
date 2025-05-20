import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
# 导入了用于处理数据、构建模型、定义损失函数和优化器的各种库，以及用于图像变换和保存的 torchvision 模块。

os.makedirs("images", exist_ok=True)




parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")     # 训练轮数（n_epochs）

parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")           # 批大小（batch_size）
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")             # 学习率（lr）

# Adam 优化器的参数（b1 和 b2）
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

# latent_dim 是生成器输入的噪声维度。
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

# img_size 和 channels 定义了生成图像的大小和通道数（这里为 28x28 的灰度图像）。
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# 定义图像形状和检查 CUDA    
img_shape = (opt.channels, opt.img_size, opt.img_size)  # 1*28*28      灰度图

cuda = True if torch.cuda.is_available() else False

# 定义生成器（Generator）
# Generator 类定义了一个生成器网络，它从随机噪声（z）中生成图像。
# block 函数是生成器的基本构建模块，包括一个全连接层、批归一化（可选）和 LeakyReLU 激活。
# 最后一个全连接层将输出调整为图像的大小，并通过 Tanh 激活函数将像素值限制在 -1 到 1 之间。
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

# Discriminator 类定义了一个判别器网络，用于判断输入图像是真实的还是生成的。
# 输入图像首先被展平为一个向量，然后通过几层全连接层和激活函数，最终输出一个概率值（通过 Sigmoid 激活），表示图像的真实性。
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity



# 初始化模型和损失函数
# 使用二分类交叉熵（BCELoss）作为损失函数。

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)

#  数据加载器
# 加载 MNIST 数据集，并进行图像大小调整和归一化，将其像素值从 [0, 1] 转换到 [-1, 1] 范围
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
# 使用 Adam 优化器分别优化生成器和判别器的参数。
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# 训练循环包括两个主要部分：训练生成器和训练判别器。
# 每个 epoch 和 batch 都会计算生成器和判别器的损失，并使用优化器更新参数。
# 生成的图像会按照 sample_interval 保存。
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):      # 遍历全部样本



        # imgs    torch.Size([64, 1, 28, 28])     tensor([[[[-1., -1., -1.,  ..., -1., -1., -1.],
        # Adversarial ground truths
        # valid 是一个包含全 1 的张量，用于标记真实图像。imgs.size(0) 代表当前 batch 的大小，valid 的形状是 (batch_size, 1)。
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)    # Variable   创建张量
        # valid     torch.Size([64, 1])
        # tensor([[1.],
        #         [1.], ...

        # fake 是一个包含全 0 的张量，用于标记生成的（虚假的）图像。 
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
        # fake  tensor([[0.],                          torch.Size([64, 1])
        #               [0.], ......

        # Configure input     # imgs    torch.Size([64, 1, 28, 28])     tensor([[[[-1., -1., -1.,  ..., -1., -1., -1.],
        # 将真实图像转换为一个 Variable，以便进行梯度计算。
        real_imgs = Variable(imgs.type(Tensor))
        # real_imgs      tensor([[[[-1., -1., -1.,  ..., -1., -1., -1.],   
        # real_imgs.shape  torch.Size([64, 1, 28, 28])


        # -----------------
        #  Train Generator
        # -----------------




        # optimizer_G.zero_grad(): 在每次训练生成器之前，先将生成器的梯度置为零
        optimizer_G.zero_grad()




        # Sample noise as generator input
        # 生成一个形状为 (batch_size, latent_dim) 的随机噪声张量 z，用于作为生成器的输入。      opt.latent_dim    100
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # z.shape     torch.Size([64, 100])          tensor([[ 0.2154,  1.0723,  0.5055,  ...,  0.2544, -0.4792,  1.9709], ...



        # Generate a batch of images
        #  使用生成器将噪声转换为生成的图像。
        gen_imgs = generator(z)   #   gen_imgs    torch.Size([64, 1, 28, 28])         tensor([[[[ 0.0406,  0.0365,  0.0144,  ...,  0.0159, -0.0038, -0.0024], ......
        # Generator(
        # (model): Sequential(
        #     (0): Linear(in_features=100, out_features=128, bias=True)
        #     (1): LeakyReLU(negative_slope=0.2, inplace=True)
        #     (2): Linear(in_features=128, out_features=256, bias=True)
        #     (3): BatchNorm1d(256, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        #     (4): LeakyReLU(negative_slope=0.2, inplace=True)
        #     (5): Linear(in_features=256, out_features=512, bias=True)
        #     (6): BatchNorm1d(512, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        #     (7): LeakyReLU(negative_slope=0.2, inplace=True)
        #     (8): Linear(in_features=512, out_features=1024, bias=True)
        #     (9): BatchNorm1d(1024, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        #     (10): LeakyReLU(negative_slope=0.2, inplace=True)
        #     (11): Linear(in_features=1024, out_features=784, bias=True)
        #     (12): Tanh()
        # )
        # )        


        # Discriminator(
        #   (model): Sequential(
        #     (0): Linear(in_features=784, out_features=512, bias=True)
        #     (1): LeakyReLU(negative_slope=0.2, inplace=True)
        #     (2): Linear(in_features=512, out_features=256, bias=True)
        #     (3): LeakyReLU(negative_slope=0.2, inplace=True)
        #     (4): Linear(in_features=256, out_features=1, bias=True)
        #     (5): Sigmoid()
        #   )
        # )
        # Loss measures generator's ability to fool the discriminator
        # 将生成的图像输入判别器，并将输出与真实标签（valid）进行比较，以计算生成器的损失。这里希望生成器能欺骗判别器，让判别器认为生成的图像是真的（标签为 1）。
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)       #  adversarial_loss = torch.nn.BCELoss()
        # discriminator(gen_imgs).shape    torch.Size([64, 1])
        # g_loss tensor(0.7147, device='cuda:0', grad_fn=<BinaryCrossEntropyBackward0>)


        g_loss.backward()       # 反向传播计算生成器的梯度
        optimizer_G.step()      # 更新生成器的参数。

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # real_imgs      tensor([[[[-1., -1., -1.,  ..., -1., -1., -1.],   
        # real_imgs.shape  torch.Size([64, 1, 28, 28])        
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)       # 判别器接收真实图像作为输入，输出应该尽量接近真实标签（valid），计算真实图像的损失。 # BCE
        # valid      torch.Size([64, 1])   # tensor([[1.], ....
        # discriminator(real_imgs)    # torch.Size([64, 1])       # tensor([[0.5254],


        # 判别器接收生成的图像作为输入，输出应该尽量接近虚假标签（fake），计算生成图像的损失。注意这里使用了 gen_imgs.detach()，这是为了防止在训练判别器时影响生成器的梯度更新。
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)   




        # 判别器的总损失是真实图像和生成图像损失的平均值。 
        d_loss = (real_loss + fake_loss) / 2




        d_loss.backward()
        optimizer_D.step()

        # 计算当前已经完成的 batch 数（batches_done）。
        # 如果完成的 batch 数是 sample_interval 的倍数，就将生成的图像保存到文件夹中。
        # save_image 会保存前 25 张生成的图像，排列成 5 行，并对图像像素值进行归一化处理。
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
