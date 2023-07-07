import json
import math
import os
import requests
import random

from datasets import load_dataset
from multiprocessing.pool import ThreadPool




class model_text_generator:
    def __init__(self, dataset_name, base_model, base_tokenizer, cache_dir, batch_size=50, text_maximum_lenth=200, text_minimum_lenth=55, prompt_tokens=30, 
                 device='cuda', openai_model=None, openai_key=None, do_top_k=False, top_k=40, do_top_p=False, top_p=0.96):
        self.dataset_name = dataset_name
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.text_maximum_lenth = text_maximum_lenth
        self.text_minimum_lenth = 30 if dataset_name in ['pubmed'] else text_minimum_lenth
        self.prompt_tokens = prompt_tokens
        self.device = device
        self.openai_model = openai_model
        self.openai_key = openai_key
        self.do_top_k = do_top_k
        self.top_k = top_k
        self.do_top_p = do_top_p
        self.top_p = top_p

    def move_base_model_to_gpu(self):
        if self.openai_model is None:
            self.base_model.to(self.device)
        print(f'BASE model has moved to GPU', flush=True)


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def load_texts(data_file):
    with open(data_file) as f:
        data = [json.loads(line)['text'] for line in f]

    return data


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


def _openai_sample(p, text_generator):
    import openai
    openai.api_key = text_generator.openai_key

    if text_generator.dataset_name != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": text_generator.openai_model, "max_tokens": 200}
    if text_generator.do_top_p:
        kwargs['top_p'] = text_generator.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, text_generator, separator):
    # encode each text as a list of token ids
    if text_generator.dataset_name == 'pubmed':
        texts = [t[:t.index(separator)] for t in texts]
        all_encoded = text_generator.base_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(text_generator.device)
    else:
        all_encoded = text_generator.base_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(text_generator.device)
        all_encoded = {key: value[:, :text_generator.prompt_tokens] for key, value in all_encoded.items()}

    if text_generator.openai_model:
        # decode the prefixes back into text
        prefixes = text_generator.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(text_generator.batch_size)

        #decoded = pool.map(_openai_sample, dataset, model_collector, prefixes)
        func = functools.partial(_openai_sample, text_generator)
        decoded = pool.map(func, prefixes)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least text_minimum_lenth words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < text_generator.text_minimum_lenth:
            if tries != 0:
                print(f'min words: {m}, needed {text_generator.text_minimum_lenth}, regenerating (try {tries})', flush=True)

            sampling_kwargs = {}
            if text_generator.do_top_p:
                sampling_kwargs['top_p'] = text_generator.top_p
            elif text_generator.do_top_k:
                sampling_kwargs['top_k'] = text_generator.top_k
            min_length = 50 if text_generator.dataset_name in ['pubmed'] else 150
            outputs = text_generator.base_model.generate(**all_encoded, min_length=min_length, max_length=text_generator.text_maximum_lenth, do_sample=True, **sampling_kwargs, pad_token_id=text_generator.base_tokenizer.eos_token_id, eos_token_id=text_generator.base_tokenizer.eos_token_id)
            decoded = text_generator.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    return decoded


def generate_model_written_dataset(raw_data, text_generator):
    data = {"original": [], "sampled": []}        

    for batch in range(math.ceil(len(raw_data) / text_generator.batch_size)): 
        print(f'Generating samples for batch {batch} of {math.ceil(len(raw_data) / text_generator.batch_size)}', flush=True)
        try:
            original_text = raw_data[batch * text_generator.batch_size:(batch + 1) * text_generator.batch_size]
        except:
            original_text = raw_data[batch * text_generator.batch_size:len(raw_data)]

        sampled_text = sample_from_model(original_text, text_generator, '<<<SEP>>>')

        for o, s in zip(original_text, sampled_text):
            if text_generator.dataset_name == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace('<<<SEP>>>', ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)
    
    # if pre_perturb_pct > 0:
    #     print(f'APPLYING {pre_perturb_pct}, {pre_perturb_span_length} PRE-PERTURBATIONS', flush=True)
    #     load_mask_model()
    #     data["sampled"] = perturb_texts(data["sampled"], model_collector, random_fills, random_fills_tokens, pre_perturb_span_length, pre_perturb_pct, chunk_size, ceil_pct=True)
    #     load_base_model()

    return data


def reload_cached_dataset(data_path):
    print(f'Loading DATASET ...', flush=True)
    try:
        with open(os.path.join(data_path, 'experiment_data.json')) as f:
            data = json.load(f)

        return data
    except:
        raise ValueError(f'Dataset does not exist.')


def construct_predefined_dataset(human_dataset, cache_dir):
    # ALL_DATASETS = ['xsum', 'squad', 'writing', 'pubmed', 'wmt16_en', 'wmt16_de']
    if human_dataset == 'xsum':
        data = load_dataset('xsum', split='train', cache_dir=cache_dir)['document']
    elif human_dataset == 'squad':
        data = load_dataset('squad', split='train', cache_dir=cache_dir)['context']
    elif human_dataset == 'pubmed':
        data = load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
        # combine question and long_answer
        data = [f'Question: {q} Answer:<<<SEP>>>{a}' for q, a in zip(data['question'], data['long_answer'])]
    elif human_dataset == 'wmt16_en':
        data = load_language('en', cache_dir)
    elif human_dataset == 'wmt16_de':
        data = load_language('de', cache_dir)
    elif human_dataset == 'writing':
        writing_path = cache_dir + '/writingPrompts'
        if os.path.isdir(writing_path):
            with open(f'{writing_path}/valid.wp_source', 'r') as f:
                prompts = f.readlines()
            with open(f'{writing_path}/valid.wp_target', 'r') as f:
                stories = f.readlines()
            
            prompts = [process_prompt(prompt) for prompt in prompts]
            joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
            data = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]
        else:
            raise ValueError(f"Dataset writing is not existed. Please download it first and save it into './cache_dir/writingPrompts' folder")
    else:
        raise ValueError(f'Dataset {human_dataset} is not included.')

    # get unique examples, strip whitespace, and remove newlines
    # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
    # then take just the examples that are <= 512 tokens (for the mask model)
    # then generate n_samples samples

    # remove duplicates from the data
    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()
    # strip whitespace around each example
    data = [x.strip() for x in data]
    # remove newlines from each example
    data = [' '.join(x.split()) for x in data]

    random.shuffle(data)

    # # try to keep only examples with > 250 words
    # if self.dataset_name in ['xsum', 'squad', 'writing']:
    #     long_data = [x for x in self.data if len(x.split()) > 250]
    #     if len(long_data) > 0:
    #         self.data = long_data

    return data
    

def construct_customized_hf_dataset(human_dataset, split, key_name, cache_dir):
    data = load_dataset(human_dataset, split=split, cache_dir=cache_dir)[key_name]

    data = list(dict.fromkeys(data)) 
    data = [x.strip() for x in data]
    data = [' '.join(x.split()) for x in data]

    return data


def load_preprocessed_dataset(human_dataset, synthetic_dataset, cache_dir, split='train', user_data_path=None):
    if human_dataset == 'open-web-text':
        original_text, sampled_text = get_gpt2_detector_dataset('webtext', synthetic_dataset, cache_dir)
    elif human_dataset == 'open-gpt-text':
        original_text, sampled_text = get_opengpttext_dataset(cache_dir)
    elif human_dataset == 'hc3-english' or human_dataset == 'hc3-chinese':
        original_text, sampled_text, questions = get_hc3_dataset(human_dataset, split, cache_dir)
    else:
        original_text, sampled_text = get_user_processed_dataset(human_dataset, synthetic_dataset, user_data_path)

    original_text = list(dict.fromkeys(original_text))
    sampled_text = list(dict.fromkeys(sampled_text)) 

    original_text = [x.strip() for x in original_text]
    sampled_text = [x.strip() for x in sampled_text]

    original_text = [' '.join(x.split()) for x in original_text]
    sampled_text = [' '.join(x.split()) for x in sampled_text]

    final_data = {'original':original_text, 'sampled':sampled_text}

    if human_dataset == 'hc3-english' or human_dataset == 'hc3-chinese':
        questions = list(dict.fromkeys(questions))
        questions = [x.strip() for x in questions]
        questions = [' '.join(x.split()) for x in questions]

        final_data = {'original':original_text, 'sampled':sampled_text, 'question':questions}

    return final_data
    
    # long_original_text = [x for x in original_text if len(x.split()) > 250]
    # long_sampled_text = [x for x in sampled_text if len(x.split()) > 250]

    # if len(long_original_text) > 0:
    #     original_text = long_original_text
    # if len(long_sampled_text) > 0:
    #     sampled_text = long_sampled_text

    # random.shuffle(original_text)
    # random.shuffle(sampled_text)


    # with open(f'./data/{dataset_name}', 'r') as f:
    #     self.data = f.readlines()


def get_gpt2_detector_dataset(*datasets, cache_dir, train_model=False):
    ALL_DATASETS = ['webtext',
        'small-117M',  'small-117M-k40',  'small-117M-nucleus',
        'medium-345M', 'medium-345M-k40', 'medium-345M-nucleus',
        'large-762M',  'large-762M-k40',  'large-762M-nucleus',
        'xl-1542M',    'xl-1542M-k40',    'xl-1542M-nucleus'
    ]

    if not train_model:
        dataset_name_list = ['test']
    else:
        dataset_name_list = ['train', 'valid', 'test']

    for ds in datasets:
        assert ds in ALL_DATASETS, "Please give a correct gpt2-detector dataset name"
        for split in dataset_name_list:
            filename = ds + '.' + split + '.jsonl'
            if not os.path.exists(os.path.join(cache_dir, filename)):
                r = requests.get('https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/' + filename, stream=True)

                with open(os.path.join(cache_dir, filename), 'wb') as f:
                    file_size = int(r.headers['content-length'])
                    chunk_size = 1000
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)

    if not train_model:
        original_text = load_texts(os.path.join(cache_dir, datasets[0] + '.' + split + '.jsonl'))
        sampled_text = load_texts(os.path.join(cache_dir, datasets[-1] + '.' + split + '.jsonl'))

        return original_text, sampled_text


def get_opengpttext_dataset(cache_dir):
    writing_path = cache_dir + '/open-gpt-text'

    human_dataset_name = 'open-web-text-final'
    synthetic_dataset_name = 'open-gpt-text-final'

    original_text = []
    sampled_text = []

    human_dataset_filename_list = os.listdir(os.path.join(writing_path, human_dataset_name))
    synthetic_dataset_filename_list = os.listdir(os.path.join(writing_path, synthetic_dataset_name))

    if os.path.isdir(writing_path):
        for i in range(len(human_dataset_filename_list)):
            original_text.extend(load_texts(os.path.join(writing_path, human_dataset_name + '/' + human_dataset_filename_list[i])))

        for j in range(len(synthetic_dataset_filename_list)):
            sampled_text.extend(load_texts(os.path.join(writing_path, synthetic_dataset_name + '/' + synthetic_dataset_filename_list[j])))
    else:
        raise ValueError(f"Dataset open-gpt-text is not existed. Please download it first and save it into './cache_dir/open-gpt-text' folder")
    
    return original_text, sampled_text


def get_hc3_dataset(dataset_name, split, cache_dir):
    if dataset_name == 'hc3-english':
        dataset = 'Hello-SimpleAI/HC3'
    else:
        dataset = 'Hello-SimpleAI/HC3-Chinese'
    all_data = load_dataset(dataset, data_files=['all.jsonl'], cache_dir=cache_dir)
    data = all_data[split]

    questions = ds['train']['question']
    original_text = ds['train']['human_answers']
    sampled_text = ds['train']['chatgpt_answers']

    return original_text, sampled_text, questions


def get_user_processed_dataset(human_dataset_name, synthetic_dataset_name, user_data_path):
    if os.path.isdir(user_data_path):
        try:
            original_text = load_texts(os.path.join(writing_path, human_dataset_name + '.jsonl'))
            sampled_text = load_texts(os.path.join(writing_path, synthetic_dataset_name + '.jsonl'))
        except:
            raise ValueError(f'Cannot load data from {user_data_path}. Please process it first.')
    else:
        raise ValueError(f'Dataset {user_data_path} is not existed. Please process it first and save it into {user_data_path} folder')

    return original_text, sampled_text








# def generate_data_customized(load_hf_dataset=True, key_name='text', split='train', human_dataset_name='', synthetic_dataset_name=''):
#     if load_hf_dataset:
#         data = load_dataset(dataset_name, split=split, cache_dir=self.cache_dir)[key_name]
#     else:
#         if dataset_name == 'gpt-2-output':
#             original_text, sampled_text = get_gpt2_output_dataset(synthetic_dataset_name, self.cache_dir)
#         elif dataset_name == 'open-gpt-text':
#             original_text, sampled_text = get_opengpttext_dataset(human_dataset_name, synthetic_dataset_name, self.cache_dir)
#         elif dataset_name == 'hc3-english' or dataset_name == 'hc3-chinese':
#             original_text, sampled_text, questions = get_hc3_dataset(self.dataset_name, split, self.cache_dir)

#         original_text = list(dict.fromkeys(original_text)) 
#         sampled_text = list(dict.fromkeys(sampled_text)) 

#         original_text = [x.strip() for x in original_text]
#         sampled_text = [x.strip() for x in sampled_text]

#         original_text = [' '.join(x.split()) for x in original_text]
#         sampled_text = [' '.join(x.split()) for x in sampled_text]
        
#         long_original_text = [x for x in original_text if len(x.split()) > 250]
#         long_sampled_text = [x for x in sampled_text if len(x.split()) > 250]

#         if len(long_original_text) > 0:
#             original_text = long_original_text
#         if len(long_sampled_text) > 0:
#             sampled_text = long_sampled_text

#         random.shuffle(original_text)
#         random.shuffle(sampled_text)
#         self.final_data = {'original':original_text, 'sampled':sampled_text}

