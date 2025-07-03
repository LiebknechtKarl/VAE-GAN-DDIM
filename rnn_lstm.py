import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import time

# ================== 数据准备 ==================
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for label, line in data_iter:
        yield tokenizer(line)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0

def collate_batch(batch):
    label_list, text_list = [], []
    for label, text in batch:
        label_list.append(torch.tensor(label_pipeline(label)))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.long)
        text_list.append(processed_text)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    return text_list, torch.tensor(label_list)

train_iter = IMDB(split='train')
train_loader = DataLoader(list(train_iter)[:1000], batch_size=32, shuffle=True, collate_fn=collate_batch)

# ================== 模型定义 ==================
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type='rnn'):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported RNN type")
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # 取最后一个时间步的输出

# ================== 训练对比函数 ==================
def train_and_evaluate(rnn_type='rnn'):
    model = SimpleRNN(len(vocab), embed_dim=64, hidden_dim=128, rnn_type=rnn_type).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"{rnn_type.upper()} → Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

# ================== 执行对比 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

train_and_evaluate('rnn')   # 测试 RNN
train_and_evaluate('lstm')  # 测试 LSTM
