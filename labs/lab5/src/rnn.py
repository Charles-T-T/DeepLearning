import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# 古诗生成
# 主函数在main.py中
# 本py文件中的代码需要进行填空

"""
测试结果：
>>> print(gen_poem("日照香炉生"))
G日照香炉生，风开古木清。何时此相见，犹作白云心。E

>>> print(gen_poem("清明时节雨"))
G清明时节雨，独自有林塘。草带青山远，烟微落日寒。E

>>> print(gen_poem("风"))
G风光随处见，风雨夜来深。今日阳城里，相逢不可闻。E

>>> print(gen_poem("花"))
G花开白露滴，花满绿萝开。此景皆如此，无心更有情。E

>>> print(gen_poem("雪"))
G雪后秋光媚，风开古木清。何时此相见，犹作白云心。E

>>> print(gen_poem("月"))
G月明寒草绿，风起竹林风。此地难相见，何人更见招。E

>>> print(gen_poem("雨"))
G雨洗青山绿，花随绿殿新。谁知旧溪月，不见落花前。E

>>> print(gen_poem("日"))
G日日千年月，山中见竹林。此时从此别，相见不知心。E

>>> print(gen_poem("朝"))
G朝来一里别，此别更何之。一路无人识，东南归不归。E

>>> print(gen_poem("三"))
G三月不可见，望乡今已微。今宵见秋月，独自有林塘。E

>>> print(gen_poem("九"))
G九陌通天苑，东风起鼓鼙。江湖春不尽，风雨夜猿飞。E
"""

# word embedding层
# 这里用于给每一个汉字字符(例如'床'，'前'等)以及特殊符号, 用向量进行表示
class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(
            -1, 1, size=(vocab_length, embedding_dim)
        )  # 初始化
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(
            torch.from_numpy(w_embeding_random_intial)
        )

    def forward(self, input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


# RNN模型
# 模型可以根据当前输入的一系列词预测下一个出现的词是什么
class RNN_model(nn.Module):
    def __init__(self, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()
        self.word_embedding_lookup = word_embedding
        self.vocab_length = (
            vocab_len  # 可选择的单词数目 或者说 word embedding层的word数目
        )
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #########################################
        # 这里你需要定义 "self.rnn_lstm"
        # 其中输入特征大小是 "word_embedding_dim"
        #    输出特征大小是 "lstm_hidden_dim"
        # 这里的LSTM应该有两层，并且输入和输出的tensor都是(batch, seq, feature)大小
        # (提示：LSTM层或许torch.nn中有对应的网络层,pytorch官方文档也许有说明)
        # 填空：
        self.rnn_lstm = nn.LSTM(
            input_size=self.word_embedding_dim,
            hidden_size=self.lstm_dim,
            num_layers=2,
            batch_first=True,
        )
        ##########################################
        self.fc = nn.Linear(self.lstm_dim, self.vocab_length)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, sentence, batch_size, is_test=False):
        batch_input = self.word_embedding_lookup(sentence).view(
            batch_size, -1, self.word_embedding_dim
        )
        ################################################
        # 这里你需要将上面的"batch_input"输入到你在rnn模型中定义的lstm层中
        # lstm的隐藏层输出应该被定义叫做变量"output", 初始的隐藏层(initial hidden state)和记忆层(initial cell state)应该是0向量.
        # 填空
        h0 = torch.zeros(2, batch_size, self.lstm_dim).to(self.device)
        c0 = torch.zeros(2, batch_size, self.lstm_dim).to(self.device)
        output, _ = self.rnn_lstm(batch_input, (h0, c0))
        ################################################
        out = output.contiguous().view(-1, self.lstm_dim)
        out = self.fc(out)  # out.size: (batch_size * sequence_length ,vocab_length)
        if is_test:
            # 测试阶段(或者说生成诗句阶段)使用
            prediction = out[-1, :].view(1, -1)
            output = prediction
        else:
            # 训练阶段使用
            output = out
        return output
