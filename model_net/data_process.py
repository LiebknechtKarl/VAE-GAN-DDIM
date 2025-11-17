import os
import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


class JsonlSequenceDataset(Dataset):

    def __init__(self, folder, seq_len):
        self.folder = folder
        self.seq_len = seq_len

        # 收集文件
        self.files = [os.path.join(folder, f)
                      for f in os.listdir(folder)
                      if f.endswith(".jsonl")]

        # 加载所有 jsonl
        self.data = []
        self.file_lengths = []
        for file in self.files:
            lines = load_jsonl(file)
            self.data.append(lines)
            self.file_lengths.append(len(lines))

        total_lines = sum(self.file_lengths)
        self.file_probs = [l / total_lines for l in self.file_lengths]

    def sample_seq(self):
        # 加权选择文件
        file_idx = np.random.choice(len(self.files), p=self.file_probs)
        file_data = self.data[file_idx]

        n = len(file_data)
        if n <= self.seq_len:
            return self.sample_seq()

        start = random.randint(0, n - self.seq_len)
        seq = file_data[start:start + self.seq_len]
        return seq

    def __len__(self):
        return 100000  # 虚拟长度，代表可以无限采样

    def __getitem__(self, idx):
        seq = self.sample_seq()

        # 将 json 转成 tensor（需根据你 json 结构改写）
        time_list = []
        bs_list = []
        servo_list = []
        for item in seq:
            time_list.append([item[1]])
            bs_list.append(list(item[3].values()))
            servo_list.append(list(item[2]))

        time_interval = [[- 1/60]] + [[time_list[i+1][0] - time_list[i][0]] for i in range(len(time_list)-1)]

# [-1/60] + [time_list[i+1] - time_list[i] for i in range(len(time_list)-1)]
        return (
            torch.tensor(time_interval, dtype=torch.float32),
            torch.tensor(bs_list, dtype=torch.float32),
            torch.tensor(servo_list, dtype=torch.float32)
        )


if __name__ == "__main__":
    # 测试数据集
    loader = torch.utils.data.DataLoader(
        JsonlSequenceDataset("bs_servo_data", seq_len=64),
        batch_size=16,
        shuffle=False
    )

    for time_interval, bs, servo in loader:
        print(time_interval.shape, bs.shape, servo.shape)
        # bs.shape = [batch, seq_len, num_bs]
        # servo.shape = [batch, seq_len, num_servo]
        break