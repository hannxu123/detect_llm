
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
    total_num = len(raw_fills[0])
    for i in range(len(text)):
        if '<extra_id_' in text[i]:
            text[i] = raw_fills[0][n]
            n = n + 1
            if n == total_num:
                text[i] = ' '
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
    fake_dataset = [XXXXXXXX] ## Our ChatGPT texts

    ## load T5 model
    model_name = 't5-large'
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    with torch.no_grad():
        all_fake_texts = []
        all_para_texts = []
        n = 0

        for j in range(len(fake_dataset)):
            print('Now processing sample ' +str(j))

            ## random sample extra_id
            texts = fake_dataset[j]
            all_fake_texts.append(texts)

            xx = texts.split(' ')
            random_idx = np.random.choice(200, 51, replace = False)
            random_idx = np.sort(random_idx)

            for i in range(random_idx.shape[0]):
                xx[random_idx[i]] =f"<extra_id_{(i)}>"
            texts = " ".join(xx)

            ## using T5 to edit
            n_expected = count_masks(texts)
            stop_id = mask_tokenizer.encode(f"<extra_id_{(n_expected - 1)}>")[0]
            tokens = mask_tokenizer([texts], return_tensors="pt", padding=True).to(mask_model.device)
            outputs = mask_model.generate(**tokens, max_length=300, do_sample=True, top_p=0.9,
                                          num_return_sequences=1, eos_token_id=stop_id)
            raw_fills = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
            final_sentence = join(raw_fills, texts)

            ## save
            all_para_texts.append(final_sentence)
            if j > 199:
                break

    ## save the data
    data = all_para_texts
    data = list(dict.fromkeys(data))
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    data = [process_spaces(x) for x in data]
    dd = {'original': all_fake_texts, 'paraphrased': data}

    with open('data/' + generator_name + '_edited', "wb") as fp1:  # Pickling
        pickle.dump(dd, fp1)
    print('Total ' + str(len(dd['edited'])), flush=True)

    with open('data_save/' + generator_name + '_edited', "wb") as fp1:  # Pickling
        pickle.dump(dd, fp1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test_decode', type=str, default= 'topk')
    parser.add_argument('--base_model_name', type = str, default= 'gpt2-xl')
    args = parser.parse_args()
    print('Now Doing Editing ', flush= True)
    print(args)
    run(**vars(args))
