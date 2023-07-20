import re
import nltk
nltk.download('punkt')
from nltk import sent_tokenize # for spliting English sentences
import json
from dataset2 import Corpus_all
import argparse
import random
import pickle


def filtering(text, indicating_words, language, verbose=False):
    '''removing sentence(s) that includes indicating words'''
    assert isinstance(text, str)
    assert isinstance(indicating_words, list)
    if language == 'en':
        sents = sent_tokenize(text)
    elif language == 'zh':
        sents = cut_sent(text)
    else:
        raise NotImplementedError
  
    filtered_sents = []
    for s in sents:
        has = False
        for k in indicating_words:
            if k in s:
                has = True
                break
        if not has:
            filtered_sents.append(s)
            
    filtered_sents = ' '.join(filtered_sents)
    
    #if verbose:
        #print(f'Original answers: {text} \nFiltered answers: {filtered_sents}\n')

    return filtered_sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h_or_c', type=str, default= 'human')
    parser.add_argument('--train_name', type = str, default= 'World')
    
    args = parser.parse_args()
    print(args)

    PATH = '/mnt/home/wangw116/amy/week5/detect/indicating_words_en_' + args.h_or_c + '.txt'
    # human indicating words (both english and chinese)
    with open(PATH, encoding='gbk') as f:
        indicating_words_human_en = [l.rstrip() for l in f]

    # chatgpt indicating words (both english and chinese)
    with open(PATH, encoding='gbk') as f:
        indicating_words_chatgpt_en = [l.rstrip() for l in f]


    real_data, fake_data = Corpus_all(train_name=args.train_name)
    random.shuffle(real_data)
    random.shuffle(fake_data)

    filtered_real_data = []
    filtered_fake_data = []

    n = 0
    for i in range(300):
        if( args.h_or_c == 'human'):
            
            filtered_real_data.append(filtering(real_data[i], indicating_words_human_en, 'en'))
            
            # if( real_data[i] !=filtered_real_data[i]):
                # print(".......................")
                # print('Original answers: ', real_data[i])
                # print(".......................")
                # print('Filtered answers: ', filtered_real_data[i])
                # n += 1
        else:
            filtered_fake_data.append(filtering(fake_data[i], indicating_words_chatgpt_en, 'en'))

    print(len(filtered_fake_data))
    print(len(filtered_real_data))

    if( args.h_or_c == 'human'):  
        dd = {'original': real_data, 'filtered': filtered_real_data}
    else:
        dd = {'original': fake_data, 'filtered': filtered_fake_data}

    with open(args.train_name +'_' + args.h_or_c + '_filter', "wb") as fp1:  # Pickling
        pickle.dump(dd, fp1)