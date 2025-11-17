from model import Mamba, ModelArgs
from transformers import AutoTokenizer

# One of:
#     'state-spaces/mamba-2.8b-slimpj'
#     'state-spaces/mamba-2.8b'
#     'state-spaces/mamba-1.4b'
#     'state-spaces/mamba-790m'
#     'state-spaces/mamba-370m'
#     'state-spaces/mamba-130m'

pretrained_model_name = 'state-spaces/mamba-370m'

'''
Mamba(
  (embedding): Embedding(50280, 1024)
  (layers): ModuleList(
    (0-47): 48 x ResidualBlock(
      (mixer): MambaBlock(
        (in_proj): Linear(in_features=1024, out_features=4096, bias=False)
        (conv1d): Conv1d(2048, 2048, kernel_size=(4,), stride=(1,), padding=(3,), groups=2048)
        (x_proj): Linear(in_features=2048, out_features=96, bias=False)
        (dt_proj): Linear(in_features=64, out_features=2048, bias=True)
        (out_proj): Linear(in_features=2048, out_features=1024, bias=False)
      )
      (norm): RMSNorm()
    )
  )
  (norm_f): RMSNorm()
  (lm_head): Linear(in_features=1024, out_features=50280, bias=False)
)
'''
model = Mamba.from_pretrained(pretrained_model_name)                    # Hugging Face Hub（或本地缓存）加载一个 预训练的 Mamba 模型，并返回一个可直接使用的模型实例。


# 加载一个与 GPT-NeoX-20B 模型兼容的分词器（Tokenizer）。
# 分词器的作用是：
# 把输入文本（字符串）转为模型可以处理的 token ID 序列；
# 把模型输出的 token ID 转回可读文本。
'''
GPTNeoXTokenizerFast(name_or_path='EleutherAI/gpt-neox-20b', vocab_size=50254, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	0: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	1: AddedToken("<|padding|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	50254: AddedToken("                        ", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	50255: AddedToken("                       ", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
    ... ... 
	50275: AddedToken("   ", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
	50276: AddedToken("  ", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
}
)
'''
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

import torch
import torch.nn.functional as F

def generate(model,                                 # Mamba( 。。。 。。。)
             tokenizer,                             # GPTNeoXTokenizerFast( 。。。 。。。)  
             prompt: str,                           # 'Mamba is the'
             n_tokens_to_gen: int = 50,
             sample: bool = True,
             top_k: int = 40):
    model.eval()
    
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids        # 'Mamba is the'  -----> tensor([[   46, 31834,   310,   253]])  46盲猜起始符
    
    for token_n in range(n_tokens_to_gen):                              # n_tokens_to_gen       50
        with torch.no_grad():
            indices_to_input = input_ids
            # indices_to_input  tensor([[46,31834,310,253]])---->next_token_logits  model(indices_to_input)[:,-1]   torch.Size([1, 50280])   tensor([[ 2.5647, -8.0005,  4.0953,  ..., -7.9371, -8.1013, -8.0619]])
            next_token_logits = model(indices_to_input)[:, -1]    # 最后一个
        
        probs = F.softmax(next_token_logits, dim=-1)        # probs.shape torch.Size([1, 50280]) tensor([[6.8661e-06, 1.7713e-10, 3.1725e-05,  ..., 1.8871e-10, 1.6015e-10,1.6657e-10]])
        (batch, vocab_size) = probs.shape
        
        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)      # 从概率分布中取出 前 top_k 个最大的概率值
            # values.shape      torch.Size([1, 40])     tensor([[0.0892, 0.0527, 0.0504, 0
            probs[probs < values[:, -1, None]] = 0        # 把所有小于这个阈值的概率清零。  
            probs = probs / probs.sum(axis=1, keepdims=True)            # 重新归一化为概率分布（总和为 1）。
        
        if sample:                          # True      # 随机采样（Sampling），从概率分布中随机选一个词：
            next_indices = torch.multinomial(probs, num_samples=1)
        else:                               # False     # 贪心解码（Greedy Decoding），选择概率最高的词：
            next_indices = torch.argmax(probs, dim=-1)[:, None]
        
        input_ids = torch.cat([input_ids, next_indices], dim=1)         # 拼接到输入序列，作为下一步的输入。
        # input_ids     tensor([[   46, 31834,   310,   253,   806]])    [tokenizer.decode(output.tolist()) for output in input_ids][0]     'Mamba is the first'
    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]
    
    return output_completions


print(generate(model, tokenizer, 'Mamba is the'))

# print(generate(model, tokenizer, 'John: Hi!\nSally:'))

# print(generate(model, tokenizer, 'The meaning of life is '))

# print(generate(model, tokenizer, 'def reverse_string('))






