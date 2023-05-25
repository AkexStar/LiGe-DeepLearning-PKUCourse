# Code Generation with Deep Learning 代码生成

Code Generation是一个以自然语言为输入，输出一个代码片段的任务。要求该输出的代码片段能够完成自然语言输入所描述的编程任务。在通常情况下，自然语言输入的长度单位是一个语句，而相应的程序输出可以是一行代码、多行代码或一个完整的方法体。

CONCODE是一个较为经典的Code Generation任务的数据集。

本次作业的要求是：以CONCODE数据集为训练集和测试集，完成一个支持程序代码生成的深度神经网络。

**一、任务数据集：**

本次作业的数据集选用CodeXGlue数据集中与代码生成相关的子数据集CONCODE，数据相关的格式、基本状况可以参考如下的链接：

https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code

**二、结果汇报**

请提供你的【程序源代码】及【模型训练介绍PPT】，其中PPT应包含以下内容：

（1）请提供你所采用的模型结构的图示及相关说明；

（2）请提供你的模型在验证数据集和测试数据集上的结果，衡量指标采用：Exact Match 和 BLEU

（3）请提供能够体现你的训练过程的Learn Curve及相关说明。

---

## ChatGPT回答版本

以下是使用PyTorch框架实现一个支持程序代码生成的深度神经网络的示例代码。代码使用CONCODE数据集进行训练，并在测试集上评估模型性能，同时衡量指标采用Exact Match和BLEU。

首先，导入所需的模块和库：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
```

接着，定义模型结构。在本示例中，我们使用了一个简单的带有注意力机制的Seq2Seq模型（encoder-decoder框架）。

```python
# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.GRU(hidden_size, hidden_size)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        """
        input_seq: 输入序列
        target_seq: 目标序列
        teacher_forcing_ratio: 训练时decoder输出作为下一个时间步的输入的概率
        """
        # encoder部分
        input_len = input_seq.shape[0]
        batch_size = input_seq.shape[1]
        encoder_outputs = torch.zeros(input_len, batch_size, self.hidden_size)
        encoder_outputs = encoder_outputs.to(device)
        encoder_hidden = torch.zeros(1, batch_size, self.hidden_size)
        encoder_hidden = encoder_hidden.to(device)
        for i in range(input_len):
            encoder_output, encoder_hidden = self.encoder(input_seq[i].unsqueeze(0), encoder_hidden)
            encoder_outputs[i] = encoder_output[0]
        
        # decoder部分
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
            # decoder训练阶段
            decoder_input = target_seq[0]
            for i in range(1, target_seq.shape[0]):
                decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden)
                attention_weights = F.softmax(self.attention(torch.cat((decoder_output[0], encoder_outputs[-1]), dim=1)), dim=1)
                context = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.permute(1, 2, 0)).squeeze(1)
                output = self.output(torch.cat((decoder_output[0], context), dim=1))
                target = target_seq[i]
                decoder_input = target
        else:
            # decoder生成阶段
            decoder_input = target_seq[0]
            outputs = []
            for i in range(1, target_seq.shape[0]):
                decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden)
                attention_weights = F.softmax(self.attention(torch.cat((decoder_output[0], encoder_outputs[-1]), dim=1)), dim=1)
                context = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.permute(1, 2, 0)).squeeze(1)
                output = self.output(torch.cat((decoder_output[0], context), dim=1))
                max_prob, max_idx = torch.max(output, dim=1)
                decoder_input = max_idx
                outputs.append(decoder_input.item())
        return outputs
```

定义数据集和数据处理函数：

```python
# 定义数据集
class CodeDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_seq = torch.from_numpy(row['input_seq']).long()
        target_seq = torch.from_numpy(row['target_seq']).long()
        return input_seq, target_seq

# 定义数据处理函数
def process_data(df, input_vocab_map, output_vocab_map):
    input_seqs = []
    output_seqs = []
    for i in range(len(df)):
        input_seq = df.iloc[i]['input_tokens']
        output_seq = df.iloc[i]['output_tokens']
        input_seq_len = len(input_seq)
        output_seq_len = len(output_seq)
        input_seq = [input_vocab_map[token] for token in input_seq]
        output_seq = [output_vocab_map[token] for token in output_seq]
        input_seq = np.array(input_seq + [0] * (MAX_INPUT_SEQ_LEN - input_seq_len))
        output_seq = np.array(output_seq + [0] * (MAX_OUTPUT_SEQ_LEN - output_seq_len))
        input_seqs.append(input_seq)
        output_seqs.append(output_seq)
    input_seqs = np.array(input_seqs)
    output_seqs = np.array(output_seqs)
    return input_seqs, output_seqs
```

接着，进行数据加载和模型训练：

```python
# 加载数据
df = pd.read_csv('concode/train.csv')
input_vocab_map = np.load('concode/input_vocab.npy', allow_pickle=True).item()
output_vocab_map = np.load('concode/output_vocab.npy', allow_pickle=True).item()
MAX_INPUT_SEQ_LEN = 200
MAX_OUTPUT_SEQ_LEN = 200
input_seqs, output_seqs = process_data(df, input_vocab_map, output_vocab_map)
x_train, x_val, y_train, y_val = train_test_split(input_seqs, output_seqs, test_size=0.1)

train_dataset = CodeDataset(pd.DataFrame({'input_seq': list(x_train), 'target_seq': list(y_train)}))
val_dataset = CodeDataset(pd.DataFrame({'input_seq': list(x_val), 'target_seq': list(y_val)}))

batch_size = 256
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# 训练模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = len(input_vocab_map)
hidden_size = 256
output_size = len(output_vocab_map)
model = Seq2Seq(input_size, hidden_size, output_size)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

num_epochs = 50
teacher_forcing_ratio = 0.5

train_losses = []
val_losses = []
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for input_seq, target_seq in train_dataloader:
        input_seq = input_seq.permute(1, 0).to(device)
        target_seq = target_seq.permute(1, 0).to(device)
        optimizer.zero_grad()
        outputs = model(input_seq, target_seq, teacher_forcing_ratio=teacher_forcing_ratio)
        outputs = torch.stack(outputs, dim=1)
        loss = criterion(outputs, target_seq)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for input_seq, target_seq in val_dataloader:
            input_seq = input_seq.permute(1, 0).to(device)
            target_seq = target_seq.permute(1, 0).to(device)
            outputs = model(input_seq, target_seq, teacher_forcing_ratio=0)
            outputs = torch.stack(outputs, dim=1)
            loss = criterion(outputs, target_seq)
            val_loss += loss.item()
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
    print('Epoch %d, Train Loss: %.4f, Val Loss: %.4f' % (epoch+1, train_loss, val_loss))
```

最后，进行模型评估：

```python
# 加载测试数据集
test_df = pd.read_csv('concode/test.csv')
test_input_seqs, test_output_seqs = process_data(test_df, input_vocab_map, output_vocab_map)
test_dataset = CodeDataset(pd.DataFrame({'input_seq': list(test_input_seqs), 'target_seq': list(test_output_seqs)}))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 评估模型
model.eval()
exact_match = 0
bleu_score = 0
with torch.no_grad():
    for input_seq, target_seq in test_dataloader:
        input_seq = input_seq.permute(1, 0).to(device)
        target_seq = target_seq.permute(1, 0).to(device)
        outputs = model(input_seq, target_seq, teacher_forcing_ratio=0)
        for i in range(outputs.shape[0]):
            # 计算exact match
            if torch.equal(outputs[i], target_seq[i]):
                exact_match += 1
            # 计算BLEU score
            reference = [output_vocab_map[token.item()] for token in target_seq[i] if token.item() != 0]
            candidate = [output_vocab_map[token.item()] for token in outputs[i] if token.item() != 0]
            bleu_score += sentence_bleu([reference], candidate)
            
exact_match /= len(test_dataset)
bleu_score /= len(test_dataset)
print('Exact Match: %.4f' % exact_match)
print('BLEU Score: %.4f' % bleu_score)
```

以上就是使用PyTorch框架实现的支持程序代码生成的深度神经网络的示例代码。你可以通过修改模型结构、超参数等来进行进一步的优化。