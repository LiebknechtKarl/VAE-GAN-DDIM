

from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum



import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model_net.data_process import JsonlSequenceDataset
from model_net.model_mamba import Mamba, ModelArgs

if __name__ == "__main__":

    # model = Mamba(
    #     ModelArgs(
    #         input_channel = 62,
    #         output_channel = 25,
    #         d_model=128,
    #         n_layer=4,
    #         vocab_size = 50280
    #     )
    # )

    # # 测试数据集
    # train_loader = torch.utils.data.DataLoader(
    #     JsonlSequenceDataset("bs_servo_data", seq_len=64),
    #     batch_size=16,
    #     shuffle=False
    # )

    # for time_interval, bs, servo in train_loader:
    #     print(time_interval.shape, bs.shape, servo.shape)

    #     input = torch.cat([time_interval, bs], dim=-1)  # 在最后一个维度拼接
    #     output = model(input)
    #     print(output.shape)  # Should be (16, 64, 25)


    #     break

    # #############################################

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    model = Mamba(
        ModelArgs(
            input_channel=62,
            output_channel=25,
            d_model=128,
            n_layer=4,
            vocab_size=50280
        )
    ).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 数据加载
    train_loader = torch.utils.data.DataLoader(
        JsonlSequenceDataset("bs_servo_data", seq_len=64),
        batch_size=16,
        shuffle=True
    )

    n_epochs = 10

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for time_interval, bs, servo in train_loader:
            time_interval = time_interval.to(device)
            bs = bs.to(device)
            servo = servo.to(device)

            # 前向
            input_tensor = torch.cat([time_interval, bs], dim=-1)  # (B, T, 62)
            output = model(input_tensor)  # (B, T, 25)

            # 计算损失
            loss = criterion(output, servo)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(f"Batch Loss: {loss.item():.6f}")

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.6f}")

