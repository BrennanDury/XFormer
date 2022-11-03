import torch
import re
from config import batch_size

max_length = 26
input_vocab_size = 203
output_vocab_size = 354

def load_data(file):
    words = re.sub("[.,]", '', open(file).read())
    unique = set(words.split())
    encode = {word: i+3 for i, word in enumerate(sorted(unique))}
    decode = {i+3: word for i, word in enumerate(sorted(unique))}
    decode[0] = '[PAD]'
    decode[1] = '[START]'
    decode[2] = '[END]'

    def encode_sentence(sentence):
        code = [1] + [encode[word] for word in sentence.split()] + [2]
        return code + [0] * (max_length - len(code))
    data = torch.tensor([encode_sentence(sentence) for sentence in words.split('\n')])

    def decode_code(code):
        sentence = ''
        for token in code:
            sentence += decode[int(token)] + ' '
        return sentence[:-1]
    return data, decode_code

def split_data(en, fr):
    n_datapoints = en.shape[0]
    split = int(n_datapoints * 0.8) // batch_size * batch_size

    x_train = en[:split]
    y_train = fr[:split]

    x_test = en[split:]
    y_test = fr[split:]
    training_set = torch.stack([x_train, y_train], -1).reshape(split//batch_size, batch_size, max_length, 2)
    return x_train, y_train, x_test, y_test, training_set

def trim(s, start=2):
    return s[start:torch.argwhere(s==2)]

def trim_list(a):
    r = []
    for a1 in a:
        t = []
        for a2 in a1:
            t.append(trim(a2, start=1).tolist())
        r.append(t)
    return r