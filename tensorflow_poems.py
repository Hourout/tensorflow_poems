import numpy as np
import pandas as pd
import linora as la
import tensorflow as tf

class Config():
    max_len = 6
    epoch = 100
    sample_per_epoch = 1000
    batch_size = 32
    learning_rate = 0.001
    poetry_file = 'data/poetry.txt'
    model_file = 'poetry_model.h5'
    sample_file = 'data/sample.csv'
    char_index = None
    index_char = None if char_index is None else {j:i for i,j in char_index.items()}
    poem = None

class TensorPoems():
    def __init__(self):
        self.Config = Config()
        if tf.io.gfile.exists(self.Config.model_file):
            self.model = tf.keras.models.load_model(self.Config.model_file)
            self.make_sample(trainable=False)
        else:
            self.make_sample(trainable=True)
            self.train()

    def make_sample(self, trainable=True):
        if tf.io.gfile.exists(self.Config.sample_file):
            tf.io.gfile.remove(self.Config.sample_file)
        else:
            tf.io.gfile.makedirs(self.Config.sample_file)
            tf.io.gfile.rmtree(self.Config.sample_file)
        t = pd.read_csv(self.Config.poetry_file, names=['poem']).poem.map(lambda x:x.strip().split(":")[1])
        t = t[t.map(lambda x:len(x))>5]
        self.Config.poem = t[t.map(lambda x:x[5] == '，')].reset_index(drop=True)
        char_count = la.text.word_count(self.Config.poem)
        [char_count.pop(i) for i in la.text.word_low_freq(char_count, 2)]
        self.Config.char_index = la.text.word_to_index(char_count)
        self.Config.index_char = {j:i for i,j in self.Config.char_index.items()}
        if trainable:
            pbar = tf.keras.utils.Progbar(len(self.Config.poem), stateful_metrics=['sample nums'])
            sample_nums = 0
            with tf.io.gfile.GFile(self.Config.sample_file, 'a+') as f:
                for r, i in enumerate(self.Config.poem):
                    for j in range(len(i)-self.Config.max_len-1):
                        f.write(str(la.text.word_index_sequence([list(i[j:j+self.Config.max_len+1])], self.Config.char_index)[0])[1:-1]+'\n')
                    sample_nums = sample_nums+j+1
                    pbar.update(r+1, values=[('sample nums', sample_nums)])
    
    def generate_sample_result(self, epoch, logs):
        '''训练过程中，每4个epoch打印出当前的学习情况'''
        if epoch % 4 != 0:
            return
        with open('out/out.txt', 'a',encoding='utf-8') as f:
            f.write('==================Epoch {}=====================\n'.format(epoch))
        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            print("------------Diversity {}--------------".format(diversity))
            generate = self.predict_random(temperature=diversity)
            print(generate)
            with open('out/out.txt', 'a',encoding='utf-8') as f:
                f.write(generate+'\n')
    
    def build_model(self):
        def embedding(shape, dtype=tf.float32):
            return tf.cast(tf.linalg.diag([1]*shape[0]), dtype=dtype)
        input_tensor = tf.keras.Input(shape=(self.Config.max_len,))
        x = tf.keras.layers.Embedding(len(self.Config.char_index), len(self.Config.char_index), embedding, trainable=False)(input_tensor)
        x = tf.keras.layers.LSTM(512, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.6)(x)
        x = tf.keras.layers.LSTM(256)(x)
        x = tf.keras.layers.Dropout(0.6)(x)
        dense = tf.keras.layers.Dense(len(self.Config.char_index), activation='softmax')(x)
        self.model = tf.keras.Model(input_tensor, dense)
        self.model.compile(loss=tf.losses.CategoricalCrossentropy(),
                           optimizer=tf.optimizers.Adam(lr=self.Config.learning_rate),
                           metrics=[tf.metrics.CategoricalAccuracy()])
    
    def train(self):
        def to_tensor(line):
            parsed_line = tf.io.decode_csv(line, [[0]]*(self.Config.max_len+1), field_delim=',')
            label = parsed_line[-1]
            del parsed_line[-1]
            return parsed_line, tf.reshape(label, [-1])
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = tf.data.TextLineDataset(self.Config.sample_file).map(
            to_tensor, AUTOTUNE).shuffle(self.Config.batch_size*10).batch(self.Config.batch_size).prefetch(AUTOTUNE)
        self.build_model()
        self.model.fit(dataset, epochs=self.Config.epoch, steps_per_epoch=self.Config.sample_per_epoch//self.Config.batch_size,
                       callbacks=[tf.keras.callbacks.ModelCheckpoint(self.Config.model_file, monitor='train_loss', verbose=0, save_best_only=True, save_weights_only=False),
                                  tf.keras.callbacks.LambdaCallback(on_epoch_end=self.generate_sample_result)])
    
    def predict_random(self, temperature=1):
        '''随机从库中选取一句开头的诗句，生成五言绝句'''
        sentence = self.Config.poem[np.random.choice(len(self.Config.poem), 1)[0]][:self.Config.max_len]
        generate = self.predict_sen(sentence, temperature=temperature)
        return generate

    def predict_first(self, char, temperature =1):
        '''根据给出的首个文字，生成五言绝句'''
        index = random.randint(0, self.poems_num)
        #选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1-self.Config.max_len:] + char
        generate = str(char)
#         print('first line = ',sentence)
        # 直接预测后面23个字符
        generate += self._preds(sentence,length=23,temperature=temperature)
        return generate
    
    def predict_sen(self, text, temperature =1):
        '''根据给出的前max_len个字，生成诗句'''
        '''此例中，即根据给出的第一句诗句（含逗号），来生成古诗'''
        assert self.Config.max_len==len(text), 'length should not be equal {}.'.format(self.Config.max_len)
        generate = str(text)
        generate += self._preds(text, length=24-self.Config.max_len, temperature=temperature)
        return generate
    
    def predict_hide(self, text,temperature = 1):
        '''根据给4个字，生成藏头诗五言绝句'''
        if len(text)!=4:
            print('藏头诗的输入必须是4个字！')
            return
        
        index = random.randint(0, self.poems_num)
        #选取随机一首诗的最后max_len字符+给出的首个文字作为初始输入
        sentence = self.poems[index][1-self.config.max_len:] + text[0]
        generate = str(text[0])
        print('first line = ',sentence)
        
        for i in range(5):
            next_char = self._pred(sentence,temperature)           
            sentence = sentence[1:] + next_char
            generate+= next_char
        for i in range(3):
            generate += text[i+1]
            sentence = sentence[1:] + text[i+1]
            for i in range(5):
                next_char = self._pred(sentence,temperature)           
                sentence = sentence[1:] + next_char
                generate+= next_char
        return generate
    
    def _preds(self, sentence, length=23, temperature=1):
        '''
        sentence:预测输入值
        lenth:预测出的字符串长度
        供类内部调用，输入max_len长度字符串，返回length长度的预测值字符串
        '''
        generate = ''
        for i in range(length):
            pred = self._pred(sentence, temperature)
            generate += pred
            sentence = sentence[1:]+pred
        return generate
        
    def _pred(self, sentence, temperature=1):
        '''内部使用方法，根据一串输入，返回单个预测字符'''
        x_pred = np.array(la.text.word_index_sequence([list(sentence)], self.Config.char_index))
        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds, temperature=temperature)
        next_char = self.Config.index_char[next_index]
        return next_char
    
    def sample(self, preds, temperature=1.0):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
        '''
        prob = np.asarray(preds).astype('float64')
        prob = np.power(prob, 1./temperature)
        prob = prob / np.sum(prob)
        prob = np.random.choice(range(len(prob)), 1, p=prob)
        return int(prob.squeeze())
    
if __name__ == '__main__':
    tp = TensorPoems()
#     tp.make_sample()
#     tp.train()
