import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import string
import time

SEQ_END_SYMBOL = "<EOS>"
UPPER_CHARS_NUMBER = 26  # 大写字母个数
EMBEDDING_LAYER_DIM = 64
GRU_HIDDEN_DIM = 100
BATCH_SIZE = 64
SEQ_LENGTH = 10  # 序列长度


# 制造数据集
def char2index(char):
    if char == SEQ_END_SYMBOL:
        return 0
    else:
        return ord(char) - ord("A") + 1


def index2char(index):
    if index == 0:
        return SEQ_END_SYMBOL
    else:
        return chr(index + ord("A") - 1)


def randomString(stringLength):
    """
    生成stringLength长度的 大写字符串(由A-Z组成)
    """
    letters = string.ascii_uppercase
    return "".join(random.choice(letters) for i in range(stringLength))


def get_batch(batch_size, length):
    """
    生成训练数据(随机length长度的大写字母序列)
    """
    batch_examples = [randomString(length) for i in range(batch_size)]
    encoder_X = [
        [char2index(char) for char in list(SampleString)]
        for SampleString in batch_examples
    ]
    Y = [list(reversed(IndexList)) for IndexList in encoder_X]
    decoder_X = [[char2index(SEQ_END_SYMBOL)] + IndexList[:-1] for IndexList in Y]
    return (
        batch_examples,
        torch.LongTensor(encoder_X),
        torch.LongTensor(Y),
        torch.LongTensor(decoder_X),
    )


class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.vocab_size = UPPER_CHARS_NUMBER + 1
        self.embedding_layer = nn.Embedding(
            self.vocab_size, embedding_dim=EMBEDDING_LAYER_DIM
        )
        #####################################################################################
        # 填空
        # 创建一个输入维度为EMBEDDING_LAYER_DIM，隐藏层维度为GRU_HIDDEN_DIM的单层单向GRU作为encoder
        self.encoder = nn.GRU(
            input_size=EMBEDDING_LAYER_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )
        #####################################################################################
        self.decoder = nn.GRU(
            input_size=EMBEDDING_LAYER_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )
        self.linear = nn.Linear(GRU_HIDDEN_DIM, self.vocab_size)

    def forward(self, encoder_X, decoder_X):
        #######################################################################################
        # encoder_X是encoder的输入序列，为(batch_size,sequence_size)的字符index Tensor
        # decoder_X是decoder的输入序列，为(batch_size,sequence_size)的字符index Tensor
        # 填空
        # 1. 使用编码器对输入的序列进行编码，得到当前的隐藏层状态(hidden_state)
        encoder_embedding = self.embedding_layer(encoder_X)
        decoder_embedding = self.embedding_layer(decoder_X)
        _, hidden_state = self.encoder(encoder_embedding)

        # 2. 使用encoder得到的隐藏层状态作为decoder的初始隐藏层状态(hidden_state)
        decoder_output, _ = self.decoder(decoder_embedding, hidden_state)

        # 根据decoder每一位的hidden state预测对应的字符可能是哪个
        logit = self.linear(decoder_output).view(-1, self.vocab_size)
        # decoder_output  size: (batch_size,sequence_length,hidden_size)
        #######################################################################################
        return logit

    def encoding(self, encoder_X):
        encoder_embedding = self.embedding_layer(
            encoder_X
        )  # (batch_size, sequence_length, embedding_size)
        encoder_output, hidden_state = self.encoder(encoder_embedding)
        # hidden_state: (整个序列输入后)当前encoder隐藏层状态
        # hidden_state shape: (1, batch_size, hidden_size)
        # encoder_out: 对于输入序列中每一位输入对应的encoder隐藏层状态
        # encoder_out shape： (batch_size, sequence_length, hidden_size)
        return encoder_output, hidden_state

    def decoding(self, decoder_X, hidden_state):
        all_outputs = []
        for i in range(SEQ_LENGTH):
            input_X = decoder_X[:, i].unsqueeze(1)  # (batch_size,1)
            input_embedding = self.embedding_layer(input_X)
            output, hidden_state = self.decoder(input_embedding, hidden_state)
            all_outputs.append(output)
        all_outputs = torch.cat(
            all_outputs, dim=1
        )  # (batch_size,sequence_length,hidden_size)
        return all_outputs, hidden_state

    def get_next_token(self, input_X, hidden_state):
        # 此函数只用于测试(生成序列)
        # 根据hidden state和当前序列的最后一个字符预测下一个字符是啥
        # input_X shape: (batch_size, 1)
        input_embedding = self.embedding_layer(input_X)
        output, new_hidden_state = self.decoder(input_embedding, hidden_state)
        output = self.linear(output.squeeze(1))
        _, output = torch.topk(output, k=1, dim=-1)  # 找到分数最高(概率最大)的字符
        return output.view(-1), new_hidden_state


def train_one_step(model, optimizer, criterion, encoder_X, Y, decoder_X):
    using_time = time.time()
    optimizer.zero_grad()
    logits = model(encoder_X, decoder_X)
    loss = criterion(logits, Y)
    loss_value = loss.item()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    using_time = time.time() - using_time
    return loss_value, using_time


def train(model, optimizer, criterion):
    for step in range(5000):
        batch_samples, encoder_X, Y, decoder_X = get_batch(BATCH_SIZE, SEQ_LENGTH)
        Y = Y.view(-1)
        loss, using_time = train_one_step(
            model, optimizer, criterion, encoder_X, Y, decoder_X
        )
        if (step + 1) % 100 == 0:
            print(
                "step {} loss: {}  -- using time {:3f}".format(step, loss, using_time)
            )


def test(model):
    test_num = 100  # 测试示例个数
    for one in range(test_num):
        # 创建一例测试用的数据
        batch_samples, encoder_X, Y, decoder_X = get_batch(1, SEQ_LENGTH)

        original_sequence = "".join(reversed(batch_samples[0]))  # 应该生成的序列
        generate_sequence = ""  # 实际生成的序列

        # 先用encoder编码
        _, hidden_state = model.encoding(encoder_X)
        current_Char = SEQ_END_SYMBOL
        current_Index = char2index(current_Char)
        # 然后用decoder解码，进行序列生成，当前用decoder生成的字符是下次decoder的输入字符
        with torch.no_grad():
            for i in range(SEQ_LENGTH):
                input_X = torch.LongTensor([[current_Index]])  # .cuda()
                output, hidden_state = model.get_next_token(input_X, hidden_state)
                current_Index = output.cpu().tolist()[0]
                current_Char = index2char(current_Index)
                generate_sequence += current_Char
        print(
            "Input: {}, Should get: {}, Model generate: {}  |The result is {}".format(
                batch_samples,
                original_sequence,
                generate_sequence,
                original_sequence == generate_sequence,
            )
        )


if __name__ == "__main__":
    seq2seq = Seq2SeqModel()
    optimizer = optim.Adam(seq2seq.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_using_time = time.time()
    train(seq2seq, optimizer, criterion)
    print("model training all using time {:.3f}".format(time.time() - train_using_time))
    test(seq2seq)

"""
输出结果前五行：
Input: ['PEZXGABZPK'], Should get: KPZBAGXZEP, Model generate: KPZBAGXZEP  |The result is True
Input: ['NRJZILURWR'], Should get: RWRULIZJRN, Model generate: RWRULIZJRN  |The result is True
Input: ['GKWYBKEZES'], Should get: SEZEKBYWKG, Model generate: SEZEKBYWKG  |The result is True
Input: ['EYKMGQTEYT'], Should get: TYETQGMKYE, Model generate: TYETQGMKYE  |The result is True
Input: ['WKHXWVPIAU'], Should get: UAIPVWXHKW, Model generate: UAIPVWXHKL  |The result is False
"""
