
import pickle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import json
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import datasets
datasets.logging.set_verbosity_error()
import pandas as pd

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def remove_last_sentence(paragraph):
    # Split the paragraph into sentences
    sentences = paragraph.split('. ')
    # Check if the last sentence ends with '.', '?', or '!'
    last_sentence = sentences[-1]
    if last_sentence.endswith(('.', '?', '!')):
        return paragraph  # Return the original paragraph if last sentence is not ended by '.', '?', or '!'
    else:
        if len(sentences) > 1:
            sentences.pop()
        # Join the remaining sentences
        modified_paragraph = '. '.join(sentences) +'.'
        return modified_paragraph

def process_spaces(story):
    story = story[0].upper() + story[1:]
    story = story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').replace(
        '<br />', '').strip()
    story = remove_last_sentence(story)
    return story

def strip_newlines(text):
    return ' '.join(text.split())

def cut_data(data):
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    data = [process_spaces(x) for x in data]
    return data


def Corpus_amazon():
    fake_test = []
    all_labels = []
    all_questions = []

    label = 0
    for p in ['p1', 'p2', 'p3']:
        label = label + 1
        file_path = 'data_save/amazon_' + p + '_results.jsonl'
        jsonl_data = load_jsonl(file_path)

        n = 0
        for message in jsonl_data:

            try:
                fake_test.append(message[1]['choices'][0]['message']['content'])
                q = message[0]['messages'][0]['content']
                if p == 'p1':
                    q = q.split('a review for the product ')[1]
                    q = q.split(' in around')[0]
                elif p == 'p2':
                    q = q.split('product review for ')[1]
                    if 'in 50 words' in q:
                        q = q.split(' in 50 words')[0]
                    else:
                        q = q.split(' in 100 words')[0]
                else:
                    q = q.split('just bought')[1]
                    q = q.split('from Amazon')[0]
                all_questions.append(q)
                all_labels.append(label)
                n = n + 1
                if n > 199:
                    break
            except:
                fake_test = fake_test
    fake_test = cut_data(fake_test)

    ## processing human data
    # real_test = []
    # real_question = []
    # topics= ['Apparel_v1_00', 'Electronics_v1_00', 'Software_v1_00', 'Sports_v1_00', 'Toys_v1_00']
    # for t in topics:
    #     dataset = load_dataset('amazon_us_reviews', t, cache_dir = 'cache/')['train']
    #     n = 0
    #     for dat in dataset:
    #         if len(dat['review_body'].split()) > 32:
    #             real_test.append(dat['review_body'])
    #             real_question.append(dat['product_title'])
    #             n = n + 1
    #             if n > 200:
    #                 print('Done collecting reviews from ' + t)
    #                 break
    #
    # dd = {'question': real_question, 'answer':real_test}
    # with open('data_save/real_amazon_survey', "wb") as fp1:  # Pickling
    #      pickle.dump(dd, fp1)
    # input(123)
    with open('data_save/real_amazon_survey', "rb") as fp:  # Unpickling
        dd = pickle.load(fp)
        real_test = dd['answer']
        real_question = dd['question']
    real_test = cut_data(real_test)

    all_labels.extend([0] * len(real_test))
    all_questions.extend(real_question)
    all_test = fake_test + real_test

    return all_questions, all_test, all_labels




def Corpus_world():
    fake_test = []
    all_labels = []
    all_questions = []

    label = 1000
    file_path = 'data_save/world' + '_results.jsonl'
    jsonl_data = load_jsonl(file_path)

    n = 0
    for message in jsonl_data:
        try:
            fake_test.append(message[1]['choices'][0]['message']['content'])
            all_questions.append('q')
            all_labels.append(label)
            n = n + 1
            if n > 199:
                break
        except:
            continue

    fake_test = cut_data(fake_test)

    ## processing human data
    with open('data_save/real_worldnews', "rb") as fp:  # Unpickling
        real_test = pickle.load(fp)

    real_test = [i[0] for i in real_test]
    real_test = cut_data(real_test)
    all_labels.extend([0] * len(real_test))
    all_questions.extend(['q'] * len(real_test))
    all_test = fake_test + real_test

    return all_questions, all_test, all_labels




def Corpus_ivy():
    fake_test = []
    all_labels = []
    all_questions = []

    label = 1000
    file_path = 'data_save/ivy_results.jsonl'
    jsonl_data = load_jsonl(file_path)

    n = 0
    for message in jsonl_data:
        try:
            fake_test.append(message[1]['choices'][0]['message']['content'])
            all_questions.append('q')
            all_labels.append(label)
            n = n + 1
            if n > 199:
                break
        except:
            continue

    fake_test = cut_data(fake_test)

    ## processing human data
    real_test = load_dataset("qwedsacf/ivypanda-essays")['train']
    real_test = [real_test[i]['TEXT'] for i in range(len(real_test)) if i < 200]
    real_test = cut_data(real_test)
    all_labels.extend([0] * len(real_test))
    all_questions.extend(['q'] * len(real_test))
    all_test = fake_test + real_test

    return all_questions, all_test, all_labels






def Corpus_eli5():
    fake_test = []
    all_labels = []
    all_questions = []

    label = 0
    for p in ['p1', 'p2', 'p3']:
        label = label + 1
        file_path = 'data_save/eli5_' + p + '_results.jsonl'
        jsonl_data = load_jsonl(file_path)

        n = 0
        for message in jsonl_data:

            try:
                fake_test.append(message[1]['choices'][0]['message']['content'])
                q = message[0]['messages'][0]['content']
                if p == 'p1':
                    if 'no more than 50 words:' in q:
                        q = q.split('no more than 50 words:')[1]
                    elif 'no more than 100 words' in q:
                        q = q.split('no more than 100 words:')[1]
                    else:
                        q = q.split('no more than 150 words:')[1]

                elif p == 'p2':
                    if 'no more than 50 words:' in q:
                        q = q.split('no more than 50 words:')[1]
                    elif 'no more than 100 words:' in q:
                        q = q.split('no more than 100 words:')[1]
                    else:
                        q = q.split('no more than 150 words:')[1]
                    q = q.split('Please explain like I\'m five.')[0]
                else:
                    q = q.split('you would to a child:')[1]
                    q = q.split('Do not include user id ')[0]
                all_questions.append(q)
                all_labels.append(label)
                n = n + 1
                if n > 199:
                    break
            except:
                fake_test = fake_test

    fake_test = cut_data(fake_test)

    ################ real data
    real_test = []
    real_questions = []
    n = 0
    data = load_dataset("eli5")['train_eli5']

    for dat in data:
        dat_list = dat['answers']['text']
        for j in dat_list:
            if (len(j.split()) > 32) & ("URL" not in j):
                real_test.append(j)
                real_questions.append(dat['title'])
                n = n + 1
                break
        if n > 600:
            break

    real_test = cut_data(real_test)

    all_test = fake_test + real_test
    all_questions = all_questions + real_questions
    all_labels.extend([0] * len(real_test))

    return all_questions, all_test, all_labels




def Corpus_imdb():
    fake_test = []
    all_labels = []
    all_questions = []

    label = 0
    for p in ['p1', 'p2', 'p3']:
        label = label + 1
        temp_data = []
        question = []
        file_path = 'data_save/imdb_' + p + '_results.jsonl'
        jsonl_data = load_jsonl(file_path)
        for message in jsonl_data:
            try:
                temp_data.append(message[1]['choices'][0]['message']['content'])
                q = message[0]['messages'][0]['content']

                if p == 'p1':
                    q = q.split('Write a review for')[1]
                    if 'in 50 words' in q:
                        q = q.split('in 50 words')[0]
                    elif 'in 100 words' in q:
                        q = q.split('in 100 words')[0]
                    else:
                        q = q.split('in 200 words')[0]
                elif p == 'p2':
                    q = q.split('engaging and creative review for')[1]
                    if 'in 50 words' in q:
                        q = q.split('in 50 words')[0]
                    elif 'in 100 words' in q:
                        q = q.split('in 100 words')[0]
                    else:
                        q = q.split('in 200 words')[0]
                else:
                    q = q.split('just watched')[1]
                    q = q.split('. It is')[0]
                question.append(q)
            except:
                temp_data = temp_data
        temp_data = cut_data(temp_data)
        fake_test.extend(temp_data[0:10])
        all_labels.extend([label] * 10)
        all_questions.extend(question[0:10])

    ## processing human data
    real_tab = pd.read_csv('./data_save/real_imdb.csv', delimiter = ',')
    question = list(real_tab['Title'])
    question = ["\"" + q + "\"" for q in question]
    real_test = list(real_tab['Review'])

    all_test = fake_test + real_test
    all_questions = all_questions + question
    all_labels.extend([0] * len(real_test))

    return all_questions, all_test, all_labels




class TextDataset(Dataset):
    def __init__(self, all_questions, all_test, all_labels):
        self.all_test = all_test
        self.all_labels = all_labels
        self.all_questions = all_questions

    def __len__(self):
        return len(self.all_test)

    def __getitem__(self, index):
        question = self.all_questions[index]
        answer = self.all_test[index]
        label = self.all_labels[index]
        return question, answer, label


def loader(name = 'imdb'):
    if name == 'amazon':
        all_questions, all_test, all_labels = Corpus_amazon()
    elif name == 'imdb':
        all_questions, all_test, all_labels = Corpus_imdb()
    elif name == 'eli5':
        all_questions, all_test, all_labels = Corpus_eli5()
    elif name == 'world':
        all_questions, all_test, all_labels = Corpus_world()
    elif name == 'ivy':
        all_questions, all_test, all_labels = Corpus_ivy()
    else:
        raise ValueError

    test_dataset = TextDataset(all_questions, all_test, all_labels)
    test_loader = DataLoader(test_dataset, 1, shuffle= True, num_workers=0)

    return test_loader