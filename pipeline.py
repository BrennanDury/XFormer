import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
import heapq
import math
from config import n_epochs, warmup_steps, eps, beta1, beta2, smoothening, base_lr, d_model, n_beams
from data_management import trim, trim_list, max_length, output_vocab_size

def learning_rate(step):
    a = step**-0.5
    b = step*warmup_steps**(-1.5)
    return base_lr * (1 / math.sqrt(d_model)) * min(a, b)

def train(model, training_set):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(beta1, beta2), eps=eps)
    i = 1
    for epoch in range(n_epochs):
        for data in training_set:
            for g in optimizer.param_groups:
                g['lr'] = learning_rate(i)
            inputs, labels = torch.clone(data[..., 0]), torch.clone(data[..., 1])
            shift = torch.roll(labels, 1)
            shift[..., 0] = 0
            optimizer.zero_grad()
            true_probabilities = F.one_hot(labels, output_vocab_size).type(torch.float64)
            true_probabilities = (1 - smoothening) * true_probabilities + (smoothening / output_vocab_size)
            output_probabilities = model(inputs, shift)

            mask = (labels != 0) & (labels != 1)
            loss = loss_fn(output_probabilities[mask], true_probabilities[mask])
            loss.backward()
            optimizer.step()
            i += 1
        torch.save(model.state_dict(), f'models/vanilla{epoch}')

def predict(model, input, b=n_beams):
    # Beam Search
    sentence = torch.zeros(max_length, dtype=torch.int64)
    sentence[0] = 0
    sentence[1] = 1
    options = [(0, -1, sentence)]
    heapq.heapify(options)
    completed = []
    next_idx = 2
    i = 0
    while len(options) > 0:
        if next_idx >= max_length:
            break
        next_options = []
        for score, _, sentence in options:
            probabilities = torch.log(torch.softmax(model(input, sentence), -1)[0])
            best_words = torch.topk(probabilities, b, dim=-1)[1][next_idx-1]
            for next_word in best_words:
                next_sentence = torch.clone(sentence)
                next_sentence[next_idx] = next_word
                next_probability = probabilities[next_idx-1, next_word]
                next_score = ((score*(next_idx-1))+next_probability)/(next_idx)
                item = (float(next_score), i, next_sentence)
                if next_word == 2:
                    heapq.heappush(completed, (-item[0], item[2]))
                    i += 1
                else:
                    heapq.heappush(next_options, item)
                    i += 1
                    if len(next_options) > b:
                        heapq.heappop(next_options)
        options = next_options
        next_idx += 1
    score, sentence = heapq.heappop(completed)
    return trim(sentence).tolist()

def test(model, x_test, y_test, n_tests=-1):
    if n_tests == -1:
        n_tests = y_test.shape[0]
    references = trim_list(y_test[:n_tests, None, :])
    predictions = [predict(model, input) for input in x_test[:n_tests]]
    return corpus_bleu(references, predictions)