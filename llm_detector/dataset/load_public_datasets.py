# functions used to load public available datasets for detection
from datasets import load_dataset
import requests
import random
from utils import load_texts


# load datasets used for training gpt2 detectors, for more information, see https://github.com/openai/gpt-2-output-dataset/tree/master/detector
def get_gpt2_detector_dataset(*datasets, cache_dir, split):
    ALL_DATASETS = ['webtext',
        'small-117M',  'small-117M-k40',  'small-117M-nucleus',
        'medium-345M', 'medium-345M-k40', 'medium-345M-nucleus',
        'large-762M',  'large-762M-k40',  'large-762M-nucleus',
        'xl-1542M',    'xl-1542M-k40',    'xl-1542M-nucleus'
    ]

    #dataset_name_list = ['train', 'valid', 'test']
    dataset_name_list = ['test']

    for ds in datasets:
        assert ds in ALL_DATASETS, 'Please give a correct gpt2-detector dataset name'
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

    original_text = load_texts(os.path.join(cache_dir, datasets[0] + '.test.jsonl'))
    sampled_text = []
    for i in range(1, len(datasets)):
        sampled_text.extend(load_texts(os.path.join(cache_dir, datasets[i] + '.test.jsonl')))

    return original_text, sampled_text


# load open-gpt-text, for more information, see https://arxiv.org/pdf/2305.07969.pdf
def get_opengpttext_dataset(cache_dir):
    human_dataset_name = 'open-web-text-final'
    synthetic_dataset_name = 'open-gpt-text-final'

    original_text = []
    sampled_text = []

    human_dataset_filename_list = os.listdir(os.path.join(cache_dir, human_dataset_name))
    synthetic_dataset_filename_list = os.listdir(os.path.join(cache_dir, synthetic_dataset_name))

    if os.path.isdir(cache_dir):
        for i in range(len(human_dataset_filename_list)):
            original_text.extend(load_texts(os.path.join(cache_dir, human_dataset_name + '/' + human_dataset_filename_list[i])))

        for j in range(len(synthetic_dataset_filename_list)):
            sampled_text.extend(load_texts(os.path.join(cache_dir, synthetic_dataset_name + '/' + synthetic_dataset_filename_list[j])))
    else:
        raise ValueError(f'Dataset open-gpt-text is not existed. Please download it first and save it into the folder')
    
    return original_text, sampled_text


# load hc3 dataset, for more information, see https://arxiv.org/pdf/2301.07597.pdf
def get_hc3_dataset(cache_dir):
    all_data = load_dataset('hello-simpleai/hc3', data_files=['all.jsonl'], cache_dir=cache_dir)
    original_text = []
    sampled_text = []

    # questions = all_data['train']['question']
    original_text_orig = all_data['train']['human_answers']
    sampled_text_orig = all_data['train']['chatgpt_answers']

    for i in range(len(original_text_orig)):
        original_text.extend(original_text_orig[i])

    for j in range(len(sampled_text_orig)):
        sampled_text.extend(sampled_text_orig[j])

    return original_text, sampled_text


# load existing public available datasets
def load_public_dataset(human_dataset, synthetic_dataset, cache_dir):
    if human_dataset == 'open-web-text':
        original_text, sampled_text = get_gpt2_detector_dataset('webtext', synthetic_dataset, cache_dir)
    elif human_dataset == 'open-gpt-text':
        original_text, sampled_text = get_opengpttext_dataset(cache_dir)
    elif human_dataset == 'hc3':
        original_text, sampled_text = get_hc3_dataset(cache_dir)

    return original_text, sampled_text

        