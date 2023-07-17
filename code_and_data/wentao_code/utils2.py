import numpy as np
import random

def strip_newlines(text):
    return ' '.join(text.split())

def balance_acc(labels, pred):
    ## positive
    acc1 = np.sum((pred == labels) * (labels == 0)) / np.sum(labels == 0)
    acc2 = np.sum((pred == labels) * (labels == 1)) / np.sum(labels == 1)
    # print(acc1, acc2)
    return (acc1 + acc2) / 2

def sentence_cut(sentence, cut_length = 125):
    count_list = [len(s.split()) for s in sentence]
    choose_list = [(i - cut_length) for i in count_list]
    start_list = [random.randint(0, i - 1) for i in choose_list]
    cut_sentence = [' '.join(s.split()[begin:]) for begin, s in zip(start_list, sentence)]
    return cut_sentence