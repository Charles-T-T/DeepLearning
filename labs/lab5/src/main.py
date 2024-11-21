import numpy as np
import collections
import torch
import torch.optim as optim
import torch.nn as nn
import rnn
import time

# 古诗生成
# rnn.py与main.py都有需要填空的内容

start_token = "G"  # 诗句开始的标志
end_token = "E"  # 诗句结束的标志
BATCH_SIZE = 32
WORD_EMBEDDING_DIM = 100
LSTM_HIDDEN_DIM = 128
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_poems(file_name):
    # 函数用于数据预处理,将诗句用数字(word_index)表示，每一个字符用一个单独的word_index表示，从0开始
    #:param file_name: 要读取的文件名
    #:return: poems_vector, word_int_map, int_word_map
    #         poems_vector  是个2维的list ,第一维度对应每句诗, 第二个维度对应着每句诗的word index
    #         word_int_map  是个字典(dict)，key为所有诗句出现的字符，value则是字符对应的word index
    #         int_word_amp  也是个字典(dict),key为所有诗句出现的字符对应的word index,value为word index对应的字符
    #
    # e.g.
    # 从文件中读取了两句诗"月明星稀，乌鹊南飞。","乌鹊南飞，月明星稀。"
    # 我们先给他们添加上诗句开始和结束的符号 start_token end_token
    # 得到 "G月明星稀，乌鹊南飞。E","G乌鹊南飞，月明星稀。E"
    #
    # 然后对每一个字符用一个index表示(从0开始)，得到word_int_map，int_word_amp
    # word_int_map ： {'，': 0, '稀': 1, '。': 2, '飞': 3, '月': 4, 'E': 5, '星': 6, '南': 7, 'G': 8, '鹊': 9, '明': 10, '乌': 11}
    # int_word_map ： {0: '，', 1: '稀', 2: '。', 3: '飞', 4: '月', 5: 'E', 6: '星', 7: '南', 8: 'G', 9: '鹊', 10: '明', 11: '乌'}
    #
    # 然后根据word_int_map把读取的诗句用word index编码
    # poems_vector ： [[8, 4, 10, 6, 1, 0, 11, 9, 7, 3, 2, 5], [8, 11, 9, 7, 3, 0, 4, 10, 6, 1, 2, 5]]
    # 对应着 "G月明星稀，乌鹊南飞。E","G乌鹊南飞，月明星稀。" 两句

    # 读取文件
    poems = []
    with open(
        file_name,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f.readlines():
            try:
                content = line.rstrip("\n")
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    ####################################################
    # 填空------------------------------------------------------
    words = [word for poem in poems for word in poem]
    words_counter = collections.Counter(words)
    words = sorted(words_counter.items(), key=lambda x: -x[1])
    words, _ = zip(*words)

    # 建立字符和索引间的映射
    word_int_map = {word: idx for idx, word in enumerate(words)}
    int_word_map = {idx: word for idx, word in enumerate(words)}

    # 将诗句编码
    poems_vector = [[word_int_map[word] for word in poem] for poem in poems]
    ####################################################
    print(np.array(poems_vector).shape)  # 应该为 (12842, 26)
    print(len(word_int_map.keys()))  # 应该为 2001
    print(len(int_word_map.keys()))  # 应该为 2001
    return poems_vector, word_int_map, int_word_map


def generate_batch(batch_size, poems_vec):
    # 用于生成batch
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = [poem_vec[:-1] for poem_vec in poems_vec[start_index:end_index]]
        y_data = [poem_vec[1:] for poem_vec in poems_vec[start_index:end_index]]
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集
    poems_vector, word_to_int, int_to_word = process_poems("./subset_poems.txt")
    # 建立word embedding层(用于根据word index得到对应word的embedding表示，或者说向量表示)
    word_embedding = rnn.word_embedding(
        vocab_length=len(word_to_int) + 1, embedding_dim=WORD_EMBEDDING_DIM
    )
    # 建立RNN模型,之前建立的word embedding层作为它的一部分
    rnn_model = rnn.RNN_model(
        vocab_len=len(word_to_int) + 1,
        word_embedding=word_embedding,
        embedding_dim=WORD_EMBEDDING_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
    ).to(my_device)
    optimizer = optim.Adam(rnn_model.parameters(), lr=3e-3)
    Criterion = nn.CrossEntropyLoss()
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # 如果你想读取之前训练得到的模型并接着进行训练，请去掉这一行前面的#号

    for epoch in range(30):
        start_time = time.time()
        np.random.shuffle(poems_vector)
        # 生成batch
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector)
        n_chunk = len(batches_inputs)
        # 对每个batch进行训练
        for batch_num in range(n_chunk):
            batch_x = batches_inputs[batch_num]
            batch_y = batches_outputs[batch_num]
            batch_x = torch.LongTensor(batch_x).to(
                my_device
            )  # batch_x size: (batch_size, sequence_length)
            batch_y = (
                torch.LongTensor(batch_y)
                .view(
                    -1,
                )
                .to(my_device)
            )  # batch_y size: (batch_size * sequence_length)
            batch_size = batch_x.size()[0]
            pre = rnn_model(batch_x, batch_size=batch_size)
            loss = Criterion(pre, batch_y)
            print(
                "epoch  ",
                epoch + 1,
                "batch number",
                batch_num,
                "loss is: ",
                loss.tolist(),
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)  # 梯度截断(按照模)
            optimizer.step()
            optimizer.zero_grad()
        torch.save(rnn_model.state_dict(), "./poem_generator_rnn")
        print("finish  save model of epoch : {}!".format(epoch + 1))
        print("epoch using time {:.3f}".format(time.time() - start_time))


# def to_word(predict, int_to_word, poem):
#    #  预测的结果转化成对应字符...输出稍微随机一点的版本
#     if (len(poem)) % 6 == 0 or len(poem) >= 6 * 4:
#         #因为是五言诗，输出符号时不随机一点
#         sample = np.argmax(predict)
#     else:
#         predict = -predict
#         samples = predict.argsort()[:3]
#         sample = samples[np.random.randint(3)]
#     return int_to_word[sample]


def to_word(predict, int_to_word, poem):
    # 预测的结果转化成对应字符
    sample = np.argmax(predict)
    return int_to_word[sample]


def gen_poem(begin_word):
    # 用于测试，或者说古诗生成
    poems_vector, word_int_map, int_word_map = process_poems("./subset_poems.txt")
    word_embedding = rnn.word_embedding(
        vocab_length=len(word_int_map) + 1, embedding_dim=WORD_EMBEDDING_DIM
    )
    rnn_model = rnn.RNN_model(
        vocab_len=len(word_int_map) + 1,
        word_embedding=word_embedding,
        embedding_dim=WORD_EMBEDDING_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
    ).to(my_device)
    rnn_model.load_state_dict(
        torch.load("./poem_generator_rnn", weights_only=True)
    )  # 读取训练得到的模型

    # 指定开始的字
    poem = start_token + begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem], dtype=np.int64)
        input = torch.from_numpy(input).to(my_device)
        output = rnn_model(input, batch_size=1, is_test=True).to("cpu")
        word = to_word(output.data.numpy()[-1], int_word_map, poem)
        poem += word
        if len(poem) > 50:
            break
    return poem


if __name__ == "__main__":
    # 开始训练
    run_training()  # 如果不是训练阶段 ，请注销这一行。 因为网络训练时间很长。

    # 开始测试，生成诗句
    print(
        gen_poem("日照香炉生")
    )  # 这里的字可以自己指定，但是必须存在于word_embedding中
    print(gen_poem("清明时节雨"))
    print(gen_poem("风"))
    print(gen_poem("花"))
    print(gen_poem("雪"))
    print(gen_poem("月"))
    print(gen_poem("雨"))
    print(gen_poem("日"))
    print(gen_poem("朝"))
    print(gen_poem("三"))
    print(gen_poem("九"))
