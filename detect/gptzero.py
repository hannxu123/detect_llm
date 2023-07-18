import os
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from dataset2 import loader
from transformers import RobertaTokenizer

# from https://github.com/Haste171/gptzero


class GPTZeroAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.gptzero.me/v2/predict'

    def text_predict(self, document):
        url = f'{self.base_url}/text'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            'document': document
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()

    def file_predict(self, file_path):
        url = f'{self.base_url}/files'
        headers = {
            'accept': 'application/json',
            'X-Api-Key': self.api_key
        }
        files = {
            'files': (os.path.basename(file_path), open(file_path, 'rb'))
        }
        response = requests.post(url, headers=headers, files=files)
        return response.json()



def cal_metrics(label, pred_label, pred_posteriors):
    if len(set(label)) < 3:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label)
        recall = recall_score(label, pred_label)
        f1 = f1_score(label, pred_label)
        auc = roc_auc_score(label, pred_posteriors)
    else:
        acc = accuracy_score(label, pred_label)
        precision = precision_score(label, pred_label, average='weighted')
        recall = recall_score(label, pred_label, average='weighted')
        f1 = f1_score(label, pred_label, average='weighted')
        auc = -1.0
        conf_m = confusion_matrix(label, pred_label)
        print(conf_m)
    return acc, precision, recall, f1, auc

def run_gptzero_experiment(data_loader, tokenizer, gptzero_api, category):
    test_pred_prob_all = []
    test_pred_all = []
    test_label_all = []

    for i, dat in enumerate(data_loader):
        test_text_init, test_label = dat
        final_text = ' '.join(test_text_init[0].split()[0:512])
        #tokened_text = tokenizer(test_text_init[0], return_tensors="pt", padding = 'max_length', max_length = 512, truncation=True)
        #final_text = tokenizer.decode(tokened_text.input_ids[0], skip_special_tokens=True)
        test_pred_prob = gptzero_api.text_predict(final_text)['documents'][0][category]
        test_pred = round(test_pred_prob)
        test_pred_prob_all.append(test_pred_prob)
        test_pred_all.append(test_pred)
        test_label_all.append(test_label.item())

    acc_test, precision_test, recall_test, f1_test, auc_test = cal_metrics(test_label_all, test_pred_all, test_pred_prob_all)

    return f1_test

    # print(
    #     f"GPTZero acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    # print(
    #     f"GPTZero acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")


def main(dataname, category='completely_generated_prob'):
    api_key = '9abf3c4dd4f849d0a71fd2824b9450c7'  # Your API Key from https://gptzero.me
    gptzero_api = GPTZeroAPI(api_key)

    #train_loader, _, valid_loader = loader(batch_size, name=train_name, prompt= prompt, verbose= True)
    _, test_loader1, _ = loader(32, name=dataname, prompt= 'p1')
    _, test_loader2, _ = loader(32, name=dataname, prompt= 'p2')
    _, test_loader3, _ = loader(32, name=dataname, prompt= 'p3')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='./cache_dir')

    f1_value1 = run_gptzero_experiment(test_loader1, tokenizer, gptzero_api, category)
    print('P1 Test f1 score ', f1_value1)
    f1_value2 = run_gptzero_experiment(test_loader2, tokenizer, gptzero_api, category)
    print('P2 Test f1 score ', f1_value2)
    f1_value3 = run_gptzero_experiment(test_loader3, tokenizer, gptzero_api, category)
    print('P3 Test f1 score ', f1_value3)


if __name__ == '__main__':
    main('reddit', 'average_generated_prob') #average_generated_prob
    

    






    
