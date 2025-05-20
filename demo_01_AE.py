import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保存图像的函数
def save_recon_images(tensor, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"recon_epoch{epoch+1}.png")
    vutils.save_image(tensor, path, nrow=8, normalize=True)
    print(f"[Saved] {path}")

# 数据加载
transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
x_test, _ = next(iter(test_loader))
x_test = x_test.to(device)

# Autoencoder 模型
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Variational Autoencoder 模型
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc21 = nn.Linear(256, 20)  # mean
        self.fc22 = nn.Linear(256, 20)  # logvar
        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h)).view(-1, 1, 28, 28)

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# # VAE loss
# def vae_loss(recon_x, x, mu, logvar):
#     BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD
def vae_loss(recon_x, x, mu, logvar):
    recon_x = recon_x.view(-1, 784)  # ← 加这一行确保 recon_x 和 x 形状一致
    x = x.view(-1, 784)              # ← 保持一致
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 初始化模型
ae = AE().to(device)
vae = VAE().to(device)
opt_ae = torch.optim.Adam(ae.parameters(), lr=1e-4)
opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# 训练 AE
for epoch in range(5):
    ae.train()
    for x, _ in train_loader:
        x = x.to(device)
        opt_ae.zero_grad()
        x_hat = ae(x)
        loss = loss_fn(x_hat, x)
        loss.backward()
        opt_ae.step()
    print(f"[AE] Epoch {epoch+1}, Loss: {loss.item():.4f}")
    ae.eval()
    with torch.no_grad():
        recon = ae(x_test).cpu()
        save_recon_images(recon, epoch, out_dir='output/ae')

# 训练 VAE
for epoch in range(5):
    vae.train()
    for x, _ in train_loader:
        x = x.to(device)
        opt_vae.zero_grad()
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        opt_vae.step()
    print(f"[VAE] Epoch {epoch+1}, Loss: {loss.item():.4f}")
    vae.eval()
    with torch.no_grad():
        recon, _, _ = vae(x_test)
        save_recon_images(recon.cpu(), epoch, out_dir='output/vae')
