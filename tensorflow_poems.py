import numpy as np
import pandas as pd
import linora as la
import tensorflow as tf

class Config():
    max_len = 6
    epoch = 10
    sample_per_epoch = 20000
    batch_size = 32
    learning_rate = 0.001
    poetry_file = 'data/poetry.txt'
    model_file = 'data/poetry_model.h5'
    sample_file = 'data/sample.csv'
    train_log_file = 'data/train_log.txt'
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
        self.Config.char_index[' '] = 0
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
    
    def _generate_sample_result(self, epoch, logs):
        '''训练过程中，每4个epoch打印出当前的学习情况'''
        if epoch % 4 == 0:
            with open(self.Config.train_log_file, 'a+',encoding='utf-8') as f:
                f.write('==================Epoch {}=====================\n'.format(epoch))
                print("\n==================Epoch {}=====================".format(epoch))
                for diversity in [0.7, 1.0, 1.3]:
                    print("------------Diversity {}--------------".format(diversity))
                    generate = self.predict(temperature=diversity)
                    print(generate)
                    f.write(generate+'\n')
    
    def build_model(self):
        input_tensor = tf.keras.Input(shape=(self.Config.max_len,))
        x = tf.keras.layers.Embedding(len(self.Config.char_index), 300)(input_tensor)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True))(x)
        x = tf.keras.layers.GRU(216)(x)
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
            return parsed_line, tf.one_hot(label, len(self.Config.char_index))
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        dataset = (tf.data.TextLineDataset(self.Config.sample_file).map(to_tensor, AUTOTUNE)
                   .shuffle(self.Config.batch_size*10).batch(self.Config.batch_size).prefetch(AUTOTUNE))
        self.build_model()
        self.model.fit(dataset, epochs=self.Config.epoch, steps_per_epoch=self.Config.sample_per_epoch//self.Config.batch_size,
                       callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=self._generate_sample_result)])
        self.model.save(self.Config.model_file)
    
    def predict(self, text=None, hide=False, temperature=1):
        assert isinstance(text, str) or text is None, '{} length should be str or None.'.format(text)
        rand_poem = self.Config.poem[np.random.choice(len(self.Config.poem), 1)[0]]
        if text is None:
            sentence = rand_poem[:self.Config.max_len]
            generate = str(sentence)
            generate += self._predict_func(sentence, length=24-self.Config.max_len, temperature=temperature)
        elif not hide:
            sentence = rand_poem[len(text)-self.Config.max_len:]+text if len(text)<self.Config.max_len else text[-self.Config.max_len:]
            generate = str(text)
            generate += self._predict_func(sentence, length=24-len(text), temperature=temperature)
        else:
            sentence = rand_poem[1-self.Config.max_len:]+text[0]
            generate = ''
            for i in range(len(text)):
                generate += text[i]
                generate += self._predict_func(sentence, 5, temperature)
                if i!= len(text)-1:
                    sentence = generate[1-self.Config.max_len:]+text[i+1]
        return generate
    
    def _predict_func(self, sentence, length, temperature):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
        '''
        generate = ''
        char = sentence
        for i in range(length):
            pred = np.array(la.text.word_index_sequence([list(char)], self.Config.char_index))
            pred = self.model.predict(pred, verbose=0)[0]
            pred = np.power(pred, 1./temperature)
            pred = pred / np.sum(pred)
            pred = np.random.choice(range(len(pred)), 1, p=pred)[0]
            pred = self.Config.index_char[pred]
            generate += pred
            char = char[1:]+pred
        return generate
    
if __name__ == '__main__':
    tp = TensorPoems()
