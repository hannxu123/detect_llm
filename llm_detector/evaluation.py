# functions used to evaluate the detection performance of methods
import math
import torch
import params
import importlib
importlib.reload(params)
from methods import roberta_classifiers
from utils import *

# --------- exp temp ---------
from exp_temp_all import *
# --------- exp temp ---------

# obtain model output logits
def get_classifier_preds(texts, classifer, tokenizer, method_name):
    tokenized_text = tokenizer(texts, padding='max_length', truncation=True, max_length=params.OTHERS_PARAMS['padding_max_length'], return_tensors='pt').to(classifer.device)
    if 't5_sentinel' in method_name:
        decoder_input_ids = torch.tensor([tokenizer.pad_token_id] * len(texts)).unsqueeze(-1).to(classifer.device)
        logits = classifer(input_ids=tokenized_text['input_ids'], decoder_input_ids=decoder_input_ids)
        logits = logits[0].squeeze(1) 
        selected_logits = logits[:, [tokenizer('positive', return_tensors='pt')['input_ids'][0][0].item(), tokenizer('negative', return_tensors='pt')['input_ids'][0][0].item()]] 
        pred_probs = torch.softmax(selected_logits, dim=-1)
        pred_probs_pos = pred_probs[:,0].detach().cpu().numpy().tolist()
    else:
        pred_probs = classifer(input_ids=tokenized_text['input_ids'], attention_mask=tokenized_text['attention_mask'])
        if method_name == 'roberta_sentinel':
            pred_probs = torch.softmax(pred_probs, dim=-1)
        else:
            pred_probs = torch.softmax(pred_probs[0], dim=-1)
        # binary output format [fake_pred, real_pred], take the probability of model-generated text (fake) as positive
        pred_probs_pos = pred_probs[:,0].detach().cpu().numpy().tolist()
    
    return pred_probs_pos


# evaluate the classifier based methods
def classifier_evaluation(original_text, sampled_text, model, method_name):
    batch_size = params.OTHERS_PARAMS['test_batch_size']
    with torch.no_grad():
        # get predictions for original text
        orig_preds = []
        for batch in range(math.ceil(len(original_text) / batch_size)): 
            try:
                batch_orig = original_text[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_orig = original_text[batch * batch_size:len(original_text)]
            
            orig_preds.extend(get_classifier_preds(batch_orig, model.classification_model, model.classification_tokenizer, method_name))
        
        # get predictions for sampled text
        sampled_preds = []
        for batch in range(math.ceil(len(sampled_text) / batch_size)):
            try:
                batch_sampled = sampled_text[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_sampled = sampled_text[batch * batch_size:len(sampled_text)]

            sampled_preds.extend(get_classifier_preds(batch_sampled, model.classification_model, model.classification_tokenizer, method_name))

    acc, precision, recall, f1, auc, tpr, fpr = calc_metrics(orig_preds, sampled_preds)
    print(f'{method_name} test Acc: {acc}, test Precision: {precision}, test Recall: {recall}, test F1: {f1}, test AUC: {auc}, test TPR: {tpr}, test FPR: {fpr}', flush=True)


# evaluate existing tools
def tool_evaluation(original_text, sampled_text, model, method_name):
    if method_name == 'gpt_zero':
        orig_preds = [model.classification_model.text_predict(' '.join(text.split()[0:params.OTHERS_PARAMS['padding_max_length']]))['documents'][0]['completely_generated_prob'] for text in original_text]
        sampled_preds = [model.classification_model.text_predict(' '.join(text.split()[0:params.OTHERS_PARAMS['padding_max_length']]))['documents'][0]['completely_generated_prob'] for text in sampled_text]
        acc, precision, recall, f1, auc, tpr, fpr = calc_metrics(orig_preds, sampled_preds)

    print(f'{method_name} test Acc: {acc}, test Precision: {precision}, test Recall: {recall}, test F1: {f1}, test AUC: {auc}, test TPR: {tpr}, test FPR: {fpr}', flush=True)


# evaluate the performance of methods
def detection_eval(data, model, method_name):
    # --------- exp temp ---------  
    #original_text, sampled_text = load_datasets(data, False)
    original_text, sampled_text = exp_loader(None, 1, name=params.EXP_TEMP_PARAMS['dataset_name'], prompt=params.EXP_TEMP_PARAMS['prompt'], padding_max_length=params.EXP_TEMP_PARAMS['padding_max_length'], train=False)
    print(f'test on {params.EXP_TEMP_PARAMS["dataset_name"]}')
    # --------- exp temp ---------  
    if method_name in params.METHOD_PARAMS['supported_pretrained_models'] or method_name == 'roberta_base' or method_name == 'roberta_large' or method_name == 'roberta_sentinel' or method_name == 't5_sentinel' or method_name == 'mpu_roberta_base' or method_name == 'mpu_roberta_large':
        classifier_evaluation(original_text, sampled_text, model, method_name)
    elif method_name in params.METHOD_PARAMS['supported_tools']:
        tool_evaluation(original_text, sampled_text, model, method_name)



