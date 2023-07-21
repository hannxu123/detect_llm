
import transformers
import argparse
from detect.dataset2 import *

def get_non_consecutive_numbers(start, end, count):
    # Generate a list of all numbers from start to end
    all_numbers = list(range(start, end + 1))

    # Create a list to store the chosen non-consecutive numbers
    chosen_numbers = []

    # Keep choosing numbers until we have the desired count
    while len(chosen_numbers) < count:
        # Randomly select a number from the remaining pool
        selected_number = random.choice(all_numbers)

        # Add the selected number to the chosen_numbers list
        chosen_numbers.append(selected_number)

        # Remove the selected number and its adjacent numbers from the pool to ensure non-consecutiveness
        all_numbers = [x for x in all_numbers if x not in range(selected_number - 1, selected_number + 2)]

    return chosen_numbers
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

        texts = "Fantasy Football is a game played by millions of football fans around the world. It’s a game of skill, strategy and luck that’s become increasingly popular over the years. In fantasy football, you create a team of real-life NFL players and compete against other teams in your league. Your team earns points based on the performance of your players in real-life NFL games. The team with the most points at the end of the fantasy football season wins the league. For those new to the game, fantasy football can seem a bit daunting. Before you begin playing, it’s important to understand the basics of the game. You need to know how to draft your players, how to manage your team, and what strategies you can use to improve your team’s chances of success. The first step of playing fantasy football is to join a league. You can join a public league or create a private one with your friends and family. Once you’ve joined your league, it’s time to draft your team. You’ll select players from the NFL player pool and allocate them to your team. Different leagues have different rules for drafting, so make sure to read the rules of your league carefully before you start drafting. Once you’ve drafted your team, it’s time to manage it. You’ll need to set your lineup each week, making sure to choose the players that have the best chance of scoring the most points. You’ll also need to monitor the performance of your players, dropping those that are underperforming and picking up new ones to replace them."
        print(texts)
        print('......................')

        sentence_list = texts.split('.')
        filled_sentence_list = []

        for i in range(len(sentence_list)):
            xx = sentence_list[i].split()

            if len(xx) > 10:
                mask_num = 3 if len(xx) > 15 else 2
                random_idx = get_non_consecutive_numbers(1, len(xx) - 1, mask_num)
                random_idx = np.sort(random_idx)
                for j in range(random_idx.shape[0]):
                    xx[random_idx[j]] = f"<extra_id_{(j)}>"
                texts = " ".join(xx) + f" <extra_id_{(j+1)}>"

                ## using T5 to edit
                n_expected = count_masks(texts)
                stop_id = mask_tokenizer.encode(f"<extra_id_{(n_expected - 1)}>")[0]
                tokens = mask_tokenizer([texts], return_tensors="pt", padding = 'max_length', max_length = 100).input_ids.to(mask_model.device)
                outputs = mask_model.generate(tokens, max_length=100, do_sample=True, top_k = 40, eos_token_id=stop_id)
                raw_fills = mask_tokenizer.batch_decode(outputs, skip_special_tokens= False)
                final_sentence = join(raw_fills, texts)
                final_sentence = final_sentence.split('<extra_id')[0]
                filled_sentence_list.append(final_sentence)
            else:
                filled_sentence_list.append(sentence_list[i])
            #
            # print(sentence_list[i])
            # print(texts)
            # print(final_sentence)
            # input(123)

        filled_sentence = '. '.join(filled_sentence_list).replace(' .', '.')
        print(filled_sentence)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test_decode', type=str, default= 'topk')
    parser.add_argument('--base_model_name', type = str, default= 'gpt2-xl')
    args = parser.parse_args()
    print('Now Doing Editing ', flush= True)
    print(args)
    run(**vars(args))
