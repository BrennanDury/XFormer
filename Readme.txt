This repository implements a Transformer of the original design specified in Attention Is All You Need.
Changes to the parameters are made to enable cheaper training. The model is trained to translate
English to French, achieving a BLEU score of 0.39 in 20 minutes of training on my laptop.

In the future, I plan on extending this repository to implement variations of the Transformer model.

Data is from https://www.kaggle.com/code/harishreddy18/english-to-french-translation/data