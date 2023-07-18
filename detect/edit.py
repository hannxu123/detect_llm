
import transformers
import argparse
from detect.dataset2 import *

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

def count_masks(texts):
    n = 0
    xx = texts.split(' ')
    for x in xx:
        if x.startswith("<extra_id_"):
            n = n+1
    return n

def join(raw_fills, text):
    raw_fills = (extract_fills(raw_fills))
    text = text.split(' ')
    n = 0
    for i in range(len(text)):
        if '<extra_id_' in text[i]:
            try:
                text[i] = raw_fills[0][n]
                n = n+1
            except:
                break

    text = " ".join(text)
    return text

def run(data_dir='data',
        base_model_name='gpt2',
        test_decode = 'topk',
        ):

    args = locals()
    generator_name = test_decode + '_' + base_model_name

    ## data loader

    ## load T5 model
    model_name = 't5-large'
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    with torch.no_grad():
        all_fake_texts = []
        all_para_texts = []
        n = 0

        texts = 'Please note: nominations for Spring 2024 and Summer 2024 will be solicited later in the fall, ' \
                'in order to allow more time for potential nominees to complete their comprehensive exams in late summer or early fall. ' \
                'Students cannot be nominated for DCF until after they have passed the comprehensive exam.'

        print(texts)
        print('......................')

        xx = texts.split()

        random_idx = np.random.choice(range(1, len(xx) - 1), 4, replace=False)
        random_idx = np.sort(random_idx)

        for i in range(random_idx.shape[0]):
            xx[random_idx[i]] = f"<extra_id_{(i)}>"
        texts = " ".join(xx)

        print(texts)
        print('......................')

        ## using T5 to edit
        n_expected = count_masks(texts)
        stop_id = mask_tokenizer.encode(f"<extra_id_{(n_expected - 1)}>")[0]
        tokens = mask_tokenizer([texts], return_tensors="pt", padding=True).to(mask_model.device)
        outputs = mask_model.generate(**tokens, max_length=500, do_sample=True, top_k=40,
                                      num_return_sequences=1, eos_token_id=stop_id)
        raw_fills = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
        final_sentence = join(raw_fills, texts)
        print(final_sentence)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test_decode', type=str, default= 'topk')
    parser.add_argument('--base_model_name', type = str, default= 'gpt2-xl')
    args = parser.parse_args()
    print('Now Doing Editing ', flush= True)
    print(args)
    run(**vars(args))
