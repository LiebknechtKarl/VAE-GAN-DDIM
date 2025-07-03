
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

if __name__ == '__main__':

    # 初始化
    batch_size = 5
    input_size = 4
    hidden_size = 16
    mem_dim = 32        # 记忆块
    seq_len = 7
    num_layers = 4
    dropout_rate = 0.0
    # model = mLSTM(input_size=input_size, hidden_size=hidden_size)
    model = mLSTM(input_size=input_size, hidden_size=hidden_size, num_layers = num_layers, dropout_rate= dropout_rate)

    # 前向输入
    x = torch.randn(batch_size, seq_len, input_size)  # 例如 shape: [batch_size, input_size]
    print('mlstm输入',x.shape)
    # y, states = model(x, states)   # 输出 y shape: [batch_size, hidden_size]
    y, states = model(x)   # 输出 y shape: [batch_size, hidden_size]

    print('mlstm输出',y.shape, states[0].shape, states[1].shape)
