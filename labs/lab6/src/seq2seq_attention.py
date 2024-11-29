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
        # 这个seq2seq模型是添加了attention机制的升级版
        # 其中attention权重的计算采取双线性模型
        # 其中decoder的输入是当前时刻t的输入字符的embedding与
        #  根据上一时刻t-1的hidden state和encoder hidden state计算得到的聚合信息向量拼接作为输入
        super(Seq2SeqModel, self).__init__()
        self.vocab_size = UPPER_CHARS_NUMBER + 1
        self.embedding_layer = nn.Embedding(
            self.vocab_size, embedding_dim=EMBEDDING_LAYER_DIM
        )
        self.encoder = nn.GRU(
            input_size=EMBEDDING_LAYER_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=EMBEDDING_LAYER_DIM
            + GRU_HIDDEN_DIM,  # 将根据attention weight计算得到的 context vector和当前输入字符的embedding拼接作为decoder输入
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )
        self.linear = nn.Linear(GRU_HIDDEN_DIM, self.vocab_size)
        self.attention_W = nn.Linear(
            GRU_HIDDEN_DIM, GRU_HIDDEN_DIM, bias=False
        )  # 双线性模型attention中的W

    def forward(self, encoder_X, decoder_X):
        encoder_output, hidden_state = self.encoding(encoder_X)
        decoder_output, decoder_hidden = self.decoding(
            encoder_output, decoder_X, hidden_state
        )
        # decoder output_size: (batch_size,sequence_length,hidden_size)
        logit = self.linear(decoder_output).view(-1, self.vocab_size)
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

    def get_context_vector(self, encoder_out, hidden_state):
        # 用于得到得到聚合信息向量
        # 先计算attention分数后，再根据attention分数得到聚合信息向量(sum(a_1 * encoder_hidden_state_1 + a_2 * encoder_hidden_state_2 + ...))
        # encoder_out: (batch_size, sequence_length, hidden_size)
        # hidden_state: (1, batch_size, hidden_size)
        weight = torch.bmm(
            self.attention_W(encoder_out), hidden_state.permute(1, 2, 0)
        )  # ( batch_size, sequence_length ,1 )
        ###################################################################
        # 填空 对计算出的attention分数进行softmax归一化(要注意对哪一维度的值进行归一化)
        weight = torch.softmax(weight, dim=1)  # (batch_size_sequence_length,1)
        ###################################################################
        context_vectors = torch.sum(
            weight * encoder_out, dim=1, keepdim=True
        )  # (batch_size, 1, hidden_size)
        return context_vectors

    def decoding(self, encoder_out, decoder_X, hidden_state):
        all_outputs = []
        for i in range(SEQ_LENGTH):
            input_X = decoder_X[:, i].unsqueeze(1)  # (batch_size,1)
            input_embedding = self.embedding_layer(input_X)  # 字符的embedding
            #####################################################################
            # 填空
            # 1. 得到聚合信息向量(用attention机制计算得到的那个)
            context_vectors = self.get_context_vector(encoder_out, hidden_state)
            # 2. 将字符的embedding与聚合信息向量进行拼接，以作为decoder的输入
            input_embedding = torch.cat([input_embedding, context_vectors], dim=-1)
            #####################################################################
            output, hidden_state = self.decoder(input_embedding, hidden_state)
            all_outputs.append(output)
        all_outputs = torch.cat(
            all_outputs, dim=1
        )  # (batch_size,sequence_length,hidden_size)
        return all_outputs, hidden_state

    def get_next_token(self, encoder_out, input_X, hidden_state):
        # 用于预测，根据encoder的状态、当前时刻的字符输入以及上一时刻得到的hidden state预测下一个字符是啥
        # input_X shape: (batch_size, 1)
        # encoder_out shape: (batch_size, sequence_length, hidden_size)
        # hidden_state shape: (1, batch_size,hidden_size)
        #####################################################################################
        # 填空
        # 可以参考一下seq2seq的get next token函数哦
        input_embedding = self.embedding_layer(input_X)
        context_vector = self.get_context_vector(encoder_out, hidden_state)
        decoder_input = torch.cat([input_embedding, context_vector], dim=-1)
        output, new_hidden_state = self.decoder(decoder_input, hidden_state)
        logit = self.linear(output).squeeze(1)
        output = torch.argmax(logit, dim=1)
        #####################################################################################
        # output size: (batch_size)  (对应根据decoder当前隐藏层状态，通过线性层分类得到的，模型认为的最有可能的输出字符的对应index)
        # new_hidden_state (1,batch_size,hidden_size) (当前隐藏层状态)
        return output, new_hidden_state


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
        encoder_out, hidden_state = model.encoding(encoder_X)
        current_Char = SEQ_END_SYMBOL
        current_Index = char2index(current_Char)
        # 然后用decoder解码，进行序列生成，当前用decoder生成的字符是下次decoder的输入字符
        with torch.no_grad():
            for i in range(SEQ_LENGTH):
                input_X = torch.LongTensor([[current_Index]])
                output, hidden_state = model.get_next_token(
                    encoder_out, input_X, hidden_state
                )
                current_Index = output.tolist()[0]
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
Input: ['GNRIVMTUWN'], Should get: NWUTMVIRNG, Model generate: NWUTMVIRNG  |The result is True
Input: ['JWNODFTIOA'], Should get: AOITFDONWJ, Model generate: AOITFDONWJ  |The result is True
Input: ['CTNPLQQRTT'], Should get: TTRQQLPNTC, Model generate: TTRQLPNTCC  |The result is False
Input: ['LASMYFKBHB'], Should get: BHBKFYMSAL, Model generate: BHBKFYMSAL  |The result is True
Input: ['KENOIVNQPF'], Should get: FPQNVIONEK, Model generate: FPQNVIONEK  |The result is True
"""
