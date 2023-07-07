import numpy as np
import random
import os
import json
import sys
import re
import subprocess
import requests
import torch
import transformers
from datasets import load_dataset
import functools
from multiprocessing.pool import ThreadPool
from sklearn.metrics import roc_curve, precision_recall_curve, auc
        

class data_preprocess:
    def __init__(self, dataset_name, cache_dir, batch_size=50, min_words=55, prompt_tokens=30):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.min_words = 30 if dataset_name in ['pubmed'] else min_words
        self.prompt_tokens = prompt_tokens


    def generate_data_predefined(self):
        # DATASET = ['xsum', 'squad', 'writing', 'pubmed', 'wmt16_en', 'wmt16_de']
        if self.dataset_name == 'xsum':
            self.data = load_dataset('xsum', split='train', cache_dir=self.cache_dir)['document']
        elif self.dataset_name == 'squad':
            self.data = load_dataset('squad', split='train', cache_dir=self.cache_dir)['context']
        elif self.dataset_name == 'pubmed':
            self.data = load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=self.cache_dir)
            # combine question and long_answer
            self.data = [f'Question: {q} Answer:<<<SEP>>>{a}' for q, a in zip(self.data['question'], self.data['long_answer'])]
        elif self.dataset_name == 'wmt16_en':
            self.data = load_language('en', self.cache_dir)
        elif self.dataset_name == 'wmt16_de':
            self.data = load_language('de', self.cache_dir)
        elif self.dataset_name == 'writing':
            writing_path = cache_dir + '/writingPrompts'
            if os.path.isdir(writing_path):
                with open(f'{writing_path}/valid.wp_source', 'r') as f:
                    prompts = f.readlines()
                with open(f'{writing_path}/valid.wp_target', 'r') as f:
                    stories = f.readlines()
                
                prompts = [self.process_prompt(prompt) for prompt in prompts]
                joined = [self.process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
                self.data = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]
            else:
                raise ValueError(f"Dataset open-gpt-text is not existed. Please download it first and save it into './cache_dir/writingPrompts' folder")
        else:
            raise ValueError(f'Dataset {self.dataset_name} is not included.')

        # get unique examples, strip whitespace, and remove newlines
        # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
        # then take just the examples that are <= 512 tokens (for the mask model)
        # then generate n_samples samples

        # remove duplicates from the data
        self.data = list(dict.fromkeys(self.data))  # deterministic, as opposed to set()

        # strip whitespace around each example
        self.data = [x.strip() for x in self.data]

        # remove newlines from each example
        self.data = [' '.join(x.split()) for x in self.data]

        # try to keep only examples with > 250 words
        if self.dataset_name in ['xsum', 'squad', 'writing']:
            long_data = [x for x in self.data if len(x.split()) > 250]
            if len(long_data) > 0:
                self.data = long_data

        random.shuffle(self.data)


    def generate_data_customized(self, load_hf_dataset=True, key_name='text', human_dataset_name='', synthetic_dataset_name=''):
        if load_hf_dataset:
            self.data = load_dataset(dataset_name, split='train', cache_dir=self.cache_dir)[key_name]
        else:
            if self.dataset_name == 'gpt-2-output':
                original_text, sampled_text = get_gpt2_output_dataset(synthetic_dataset_name, self.cache_dir)
            elif self.dataset_name == 'open-gpt-text':
                original_text, sampled_text = get_opengpttext_dataset(human_dataset_name, synthetic_dataset_name, self.cache_dir)
            elif self.dataset_name == 'hc3-english':
                original_text, sampled_text, questions = get_hc3_dataset('hc3-english', self.cache_dir)

            # original_text = list(dict.fromkeys(original_text)) 
            # sampled_text = list(dict.fromkeys(sampled_text)) 

            # original_text = [x.strip() for x in original_text]
            # sampled_text = [x.strip() for x in sampled_text]

            # original_text = [' '.join(x.split()) for x in original_text]
            # sampled_text = [' '.join(x.split()) for x in sampled_text]
            
            # long_original_text = [x for x in original_text if len(x.split()) > 250]
            # long_sampled_text = [x for x in sampled_text if len(x.split()) > 250]

            # if len(long_original_text) > 0:
            #     original_text = long_original_text
            # if len(long_sampled_text) > 0:
            #     sampled_text = long_sampled_text

            # random.shuffle(original_text)
            # random.shuffle(sampled_text)
            # self.final_data = {"original":original_text, "sampled":sampled_text}
            # self.data = None
            if self.dataset_name == 'hc3-english':
                new_original_text = []
                new_sampled_text = []
                new_questions = []
                for i in range(len(questions)):
                    if len(sampled_text[i]) != 0:
                        new_original_text.append(original_text[i])
                        new_sampled_text.append(sampled_text[i])
                        new_questions.append(questions[i])

                new_original_text = [x[0].strip() for x in new_original_text]
                new_sampled_text = [x[0].strip() for x in new_sampled_text]
                new_questions = [x.strip() for x in new_questions]

                new_original_text = [' '.join(x.split()) for x in new_original_text]
                new_sampled_text = [' '.join(x.split()) for x in new_sampled_text]
                new_questions = [' '.join(x.split()) for x in new_questions]

                self.final_data = {'original':new_original_text, 'sampled':new_sampled_text, 'question':new_questions}
            else:
                original_text = list(dict.fromkeys(original_text))
                sampled_text = list(dict.fromkeys(sampled_text)) 

                original_text = [x.strip() for x in original_text]
                sampled_text = [x.strip() for x in sampled_text]

                original_text = [' '.join(x.split()) for x in original_text]
                sampled_text = [' '.join(x.split()) for x in sampled_text]
     
                self.final_data = {"original":original_text, "sampled":sampled_text}
            self.data = None




            # with open(f'./data/{dataset_name}', 'r') as f:
            #     self.data = f.readlines()


def load_texts(data_file, expected_size):
    with open(data_file) as f:
        data = [json.loads(line)['text'] for line in f]

    if expected_size == "ALL":
        return data
    else:
        return data[:expected_size]


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
        assert ds in ALL_DATASETS, "Please give a correct gpt2 output dataset name"
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
        original_text = load_texts(os.path.join(cache_dir, datasets[0] + '.' + split + '.jsonl'), "ALL")
        sampled_text = load_texts(os.path.join(cache_dir, datasets[-1] + '.' + split + '.jsonl'), "ALL")

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


def get_hc3_dataset(dataset_name, cache_dir):
    if dataset_name == 'hc3-english':
        dataset = 'hello-simpleai/hc3'
    else:
        dataset = 'hello-simpleai/hc3-chinese'
    ds = load_dataset(dataset, data_files=['all.jsonl'], cache_dir=cache_dir)
    #data = all_data[split]

    questions = ds['train']['question']
    original_text = ds['train']['human_answers']
    sampled_text = ds['train']['chatgpt_answers']

    return original_text, sampled_text, questions

    

def load_base_model_and_tokenizer(base_model_name, openai_model, dataset_name, cache_dir):
    if openai_model is None:
        print(f'Loading BASE model {base_model_name}...')
        base_model_kwargs = {}
        if 'gpt-j' in base_model_name or 'neox' in base_model_name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in base_model_name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in base_model_name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if dataset_name in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def generate_samples(raw_data, dataset, model_collector, dataset_name):
    data = {
        "original": [],
        "sampled": [],
    }        

    for batch in range(len(raw_data) // dataset.batch_size):
        print(f'Generating samples for batch {batch} of {len(raw_data) // dataset.batch_size}', flush=True)
        original_text = raw_data[batch * dataset.batch_size:(batch + 1) * dataset.batch_size]
        sampled_text = sample_from_model(dataset, original_text, model_collector, '<<<SEP>>>')

        for o, s in zip(original_text, sampled_text):
            if dataset_name == 'pubmed':
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


def load_language(language, cache_dir):
        # load either the english or german portion of the wmt16 dataset
        assert language in ['en', 'de']
        d = load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
        docs = d['translation']
        desired_language_docs = [d[language] for d in docs]
        lens = [len(d.split()) for d in desired_language_docs]
        sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
        return sub


def process_spaces(story):
        return story.replace(
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
            '\n ', '\n').strip()


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


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


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(dataset, texts, model_collector, separator):
    # encode each text as a list of token ids
    if dataset.dataset_name == 'pubmed':
        texts = [t[:t.index(separator)] for t in texts]
        all_encoded = model_collector.base_tokenizer(texts, return_tensors="pt", padding=True).to(model_collector.device)
    else:
        all_encoded = model_collector.base_tokenizer(texts, return_tensors="pt", padding=True).to(model_collector.device)
        all_encoded = {key: value[:, :dataset.prompt_tokens] for key, value in all_encoded.items()}

    if model_collector.openai_model:
        # decode the prefixes back into text
        prefixes = model_collector.base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(dataset.batch_size)

        #decoded = pool.map(_openai_sample, dataset, model_collector, prefixes)
        func = functools.partial(_openai_sample, dataset=dataset, model_collector=model_collector)
        decoded = pool.map(func, prefixes)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < dataset.min_words:
            if tries != 0:
                print(f'min words: {m}, needed {dataset.min_words}, regenerating (try {tries})', flush=True)

            sampling_kwargs = {}
            if model_collector.do_top_p:
                sampling_kwargs['top_p'] = model_collector.top_p
            elif model_collector.do_top_k:
                sampling_kwargs['top_k'] = model_collector.top_k
            min_length = 50 if dataset.dataset_name in ['pubmed'] else 150
            outputs = model_collector.base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=model_collector.base_tokenizer.eos_token_id, eos_token_id=model_collector.base_tokenizer.eos_token_id)
            decoded = model_collector.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    return decoded


def _openai_sample(p, dataset, model_collector):
    import openai
    openai.api_key = model_collector.openai_key

    if dataset.dataset_name != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": model_collector.openai_model, "max_tokens": 200 }
    if model_collector.do_top_p:
        kwargs['top_p'] = model_collector.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def tokenize_and_mask(text, span_length, pct, ceil_pct=False, buffer_size=1):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(model_collector, texts, mask_top_p=1.0):
    n_expected = count_masks(texts)
    stop_id = model_collector.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = model_collector.mask_tokenizer(texts, return_tensors="pt", padding=True).to(model_collector.mask_model.device)
    outputs = model_collector.mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return model_collector.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # define regex to match all <extra_id_*> tokens, where * is an integer
    pattern = re.compile(r"<extra_id_\d+>")

    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


# Get the log likelihood of each text under the base_model
def get_ll(text, model_collector):
    if model_collector.openai_model:    
        import openai 
        openai.api_key = model_collector.openai_key  
        kwargs = { "engine": model_collector.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = model_collector.base_tokenizer(text, return_tensors="pt").to(model_collector.base_model.device)
            labels = tokenized.input_ids
            return -model_collector.base_model(**tokenized, labels=labels).loss.item()


def get_lls(texts, model_collector, batch_size):
    if not model_collector.openai_model:
        return [get_ll(text, model_collector) for text in texts]
    else:
        # global API_TOKEN_COUNTER

        # # use GPT2_TOKENIZER to get total number of tokens
        # total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        # API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(batch_size)
        func = functools.partial(get_ll, model_collector=model_collector)
        return pool.map(func, texts)
        #return pool.map(get_ll, texts)


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, model_collector, log=False):
    assert args.openai_model is None, "get_rank not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = model_collector.base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = model_collector.base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text, model_collector):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = model_collector.base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = model_collector.base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

