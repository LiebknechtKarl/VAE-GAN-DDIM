import torch
from torch.utils.data import Dataset, DataLoader
import os
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

class IMDBDataset(Dataset):
    def __init__(self, dir_path, vocab=None, max_len=200):
        self.texts, self.labels = [], []
        for label, folder in enumerate(["neg", "pos"]):
            folder_path = os.path.join(dir_path, folder)
            for filename in os.listdir(folder_path):
                with open(os.path.join(folder_path, filename), encoding="utf8") as f:
                    text = clean_text(f.read())
                    self.texts.append(text)
                    self.labels.append(label)
        
        self.tokenizer = lambda x: x.split()
        self.max_len = max_len
        
        if vocab is None:
            all_tokens = [token for text in self.texts for token in self.tokenizer(text)]
            vocab = {"<pad>": 0, "<unk>": 1}
            for token in set(all_tokens):
                vocab[token] = len(vocab)
        
        self.vocab = vocab
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text)
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens[:self.max_len]]
        pad_len = self.max_len - len(ids)
        ids += [0] * pad_len
        return torch.tensor(ids), torch.tensor(self.labels[idx])




import torch.nn as nn

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, model_type='rnn'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if model_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("model_type must be 'rnn' or 'lstm'")
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out


# 第三步：定义 RNN 和 LSTM 模型
import torch.nn as nn


import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as p
class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1 , dropout_rate = 0):

        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)  # Dropout层

        self.linear_input = nn.Linear(input_size, hidden_size)

        self.W_q = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_layers)])
        self.W_k = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_layers)])
        self.W_v = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_layers)])
        self.W_i = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(self.num_layers)])
        self.W_f = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(self.num_layers)])
        self.W_o = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.num_layers)])

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def computing_unit(self, x_input, states) :
        C_prev, n_prev = states

        C_deep = []
        n_deep = []
        h_deep = []
        for layer_num in range(self.num_layers) :
            qt = self.W_q[layer_num](x_input)                     # (batch_size, hidden_size)       x_input.shape     torch.Size([5, 4])
            kt = self.W_k[layer_num](x_input) / math.sqrt(self.hidden_size)  # (batch_size, hidden_size)
            vt = self.W_v[layer_num](x_input)                     # (batch_size, hidden_size)
            # qt.shape, kt.shape, vt.shape
            # (torch.Size([5, 7, 16]), torch.Size([5, 7, 16]), torch.Size([5, 7, 16]))

            it = torch.exp(self.W_i[layer_num](x_input))          # (batch_size, 1)
            ft = torch.sigmoid(self.W_f[layer_num](x_input))      # (batch_size, 1)

            vt = vt.unsqueeze(2)                      # (batch_size, hidden_size, 1)
            kt_unsq = kt.unsqueeze(1)                 # (batch_size, 1, hidden_size)

            # 外积计算 C
            vt_kt = torch.bmm(vt, kt_unsq)            # (batch_size, hidden_size, hidden_size)

            C = ft.view(-1, 1, 1) * C_prev[:, layer_num] + it.view(-1, 1, 1) * vt_kt
            n = ft.view(-1, 1, 1) * n_prev[:, layer_num] + it.view(-1, 1, 1) * kt.unsqueeze(2)  # (B, hidden_size, 1)

            nT_qt = torch.bmm(n.transpose(1, 2), qt.unsqueeze(2)).squeeze(2)  # (batch_size, 1)
            max_nqt = torch.max(torch.abs(nT_qt), torch.tensor(1.0).to(x_input.device))

            C_qt = torch.bmm(C, qt.unsqueeze(2)).squeeze(2)   # (batch_size, hidden_size)
            h_tilde = C_qt / max_nqt                          # (batch_size, hidden_size)

            ot = torch.sigmoid(self.W_o[layer_num](x_input))              # (batch_size, hidden_size)
            ht = ot * h_tilde                                 # (batch_size, hidden_size)

            # ## 迭代
            # x_input = ht
            x_input = self.dropout(ht)
            C_deep.append(C)
            n_deep.append(n)
            h_deep.append(ht)

        # return h_deep, (C_deep, n_deep)   # torch.stack(h_list, dim=1)
        return torch.stack(h_deep, dim=1), (torch.stack(C_deep, dim=1), torch.stack(n_deep, dim=1))   # torch.stack(h_list, dim=1)

    def forward(self, x, states = None):
        """
        x: (batch_size, input_size)
        C_prev: (batch_size, hidden_size, hidden_size)
        n_prev: (batch_size, hidden_size, 1)
        """
        batch_size = x.shape[0]
        if states == None :
            states = self.init_hidden(batch_size)
        h_list = []
        C_list = []
        n_list = []

        x_ = self.linear_input(x)
        for i in range(x.shape[1]):
            ht_, (C_, n_) = self.computing_unit(x_[:,i,:],states)
            states = (C_, n_)

            h_list.append(ht_)
            C_list.append(C_)
            n_list.append(n_)

        hidden_state = torch.stack(h_list, dim=1)
        cell_state = torch.stack(C_list, dim=1)
        normalizer_state = torch.stack(n_list, dim=1)
        
        # return hidden_state, (cell_state, normalizer_state)
        return hidden_state[:,:,-1], (cell_state[:,:,-1], normalizer_state[:,:,-1])

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.num_layers, self.hidden_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.num_layers, self.hidden_size, 1, device=device))


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, model_type='rnn'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # if model_type == 'rnn':
        #     self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # elif model_type == 'lstm':
        #     self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # elif model_type == 'mlstm':         # mlstm
        #     self.rnn = mLSTM(embed_dim, hidden_dim)            
        # else:

        drop_rate = 0.2
        number_layers = 2

        if model_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, dropout=drop_rate, num_layers= number_layers, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, dropout=drop_rate,num_layers= number_layers, batch_first=True)

        elif model_type == 'mlstm':         # mlstm
            self.rnn = mLSTM(embed_dim, hidden_dim, dropout=drop_rate, num_layers= number_layers)            
        else:




            raise ValueError("model_type must be 'rnn' or 'lstm'")
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out
    


# 第四步：训练与评估函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total, correct = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()
    print(f"Train Accuracy: {correct / total:.4f}")

def test(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
    print(f"Test Accuracy: {correct / total:.4f}")

if __name__ == '__main__':


    total_epoch =10


    # 构建数据集
    train_dataset = IMDBDataset("aclImdb/train")
    test_dataset = IMDBDataset("aclImdb/test", vocab=train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)


    # 第五步：运行对比 RNN 和 LSTM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # RNN 模型
    rnn_model = TextRNN(len(train_dataset.vocab), model_type='rnn').to(device)
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("📌 训练 RNN")
    for epoch in range(total_epoch):
        train(rnn_model, train_loader, optimizer, criterion)
    test(rnn_model, test_loader)

    # LSTM 模型
    lstm_model = TextRNN(len(train_dataset.vocab), model_type='lstm').to(device)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

    print("\n📌 训练 LSTM")
    for epoch in range(total_epoch):
        train(lstm_model, train_loader, optimizer, criterion)
    test(lstm_model, test_loader)


    # mLSTM 模型
    mlstm_model = TextRNN(len(train_dataset.vocab), model_type='lstm').to(device)
    optimizer = torch.optim.Adam(mlstm_model.parameters(), lr=1e-3)

    print("\n📌 训练 mLSTM")
    for epoch in range(total_epoch):
        train(mlstm_model, train_loader, optimizer, criterion)
    test(mlstm_model, test_loader)
