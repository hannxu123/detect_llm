
import transformers
import argparse
from dataset2 import *

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
        train_name = 'world',
        h_or_c = 'human'
        ):

    args = locals()
    generator_name = test_decode + '_' + base_model_name

    ## data loader
    real_dataset, fake_dataset = Corpus_all(train_name=train_name)

    if( h_or_c == 'human'):
        x = real_dataset
    else:
        x = fake_dataset

    ## load T5 model
    model_name = 't5-large'
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    with torch.no_grad():

        all_fake_texts = []
        all_para_texts = []
        n = 0

        for j in range(len(x)):
            print('Now processing sample ' +str(j))

            ## random sample extra_id
            texts = x[j]
            all_fake_texts.append(texts)

            xx = texts.split()
            random_idx = np.random.choice(range(1, len(xx) - 1), len(xx)//2, replace=False)
            random_idx = np.sort(random_idx)

            for i in range(random_idx.shape[0]):
                xx[random_idx[i]] = f"<extra_id_{(i)}>"
            texts = " ".join(xx) + f" <extra_id_{(i+1)}>"

            ## using T5 to edit
            n_expected = count_masks(texts)
            stop_id = mask_tokenizer.encode(f"<extra_id_{(n_expected - 1)}>")[0]
            tokens = mask_tokenizer([texts], return_tensors="pt", padding=True).to(mask_model.device)
            outputs = mask_model.generate(**tokens, max_length=500, do_sample=True, top_k=40,
                                        num_return_sequences=1, eos_token_id=stop_id)
            raw_fills = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
            final_sentence = join(raw_fills, texts)
            final_sentence = final_sentence.split('<extra_id')[0]

            all_para_texts.append(final_sentence)

            if j > 298:
                break

    data = all_para_texts
    data = list(dict.fromkeys(data))
    data = [x.strip() for x in data]
    data = [strip_newlines(x) for x in data]
    data = [process_spaces(x) for x in data]
    dd = {'original': all_fake_texts, 'edited': data}

    with open("train_name + '_' + h_or_c + '_0.5_edit', "wb") as fp1:  # Pickling
        pickle.dump(dd, fp1)

    print('Total ' + str(len(dd['original'])), flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test_decode', type=str, default= 'topk')
    parser.add_argument('--base_model_name', type = str, default= 'gpt2-xl')
    parser.add_argument('--train_name', type = str, default= 'Ivypanda')
    parser.add_argument('--h_or_c', type=str, default= 'chatgpt')
    args = parser.parse_args()
    print('Now Doing Editing ', flush= True)
    print(args)
    run(**vars(args))
