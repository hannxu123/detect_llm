# functions used to evaluate the detection performance of methods
import params
import importlib
importlib.reload(params)
from methods import roberta_classifiers
from utils import *


# evaluate the classifier based methods
def classifier_evaluation(original_text, sampled_text, model, method_name):
    with torch.no_grad():
        # get predictions for original text
        orig_preds = []
        for batch in range(math.ceil(len(original_text) / batch_size)): 
            try:
                batch_real = original_text[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_real = original_text[batch * batch_size:len(original_text)]
            tokenized_batch_real = model.classification_tokenizer(batch_real, padding='max_length', truncation=True, max_length=params.OTHERS_PARAMS['padding_max_length'], return_tensors='pt').to(model.classification_model.device)
            pred_probs = model.classification_model(input_ids=tokenized_batch_real['input_ids'], attention_mask=tokenized_batch_real['attention_mask'])
            pred_probs = torch.softmax(pred_probs[0], dim=-1)
            # binary output format [fake_pred, real_pred], take the probability of model-generated text (fake) as positive
            orig_preds.extend(pred_probs[:,0].detach().cpu().numpy().tolist())
        
        # get predictions for sampled text
        sampled_preds = []
        for batch in range(math.ceil(len(sampled_text) / batch_size)):
            try:
                batch_fake = sampled_text[batch * batch_size:(batch + 1) * batch_size]
            except:
                batch_fake = sampled_text[batch * batch_size:len(sampled_text)]
            tokenized_batch_fake = model.classification_tokenizer(batch_fake, padding='max_length', truncation=True, max_length=params.OTHERS_PARAMS['padding_max_length'], return_tensors='pt').to(model.classification_model.device)
            pred_probs = model.classification_model(input_ids=tokenized_batch_fake['input_ids'], attention_mask=tokenized_batch_fake['attention_mask'])
            pred_probs = torch.softmax(pred_probs[0], dim=-1)

            sampled_preds.extend(pred_probs[:,0].detach().cpu().numpy().tolist())

    acc, precision, recall, f1, auc = calc_metrics(orig_preds, sampled_preds)
    print(f'{method_name} test Acc: {acc}, test Precision: {precision}, test Recall: {recall}, test F1: {f1}, test AUC: {auc}', flush=True)


# evaluate existing tools
def tool_evaluation(original_text, sampled_text, model, method_name):
    if method_name == 'gpt_zero':
        train_pred_prob = [model.classification_model.text_predict(text)['documents'][0]['completely_generated_prob'] for text in original_text]
        test_pred_prob = [model.classification_model.text_predict(text)['documents'][0]['completely_generated_prob'] for text in original_text]
        train_pred = [round(_) for _ in train_pred_prob]
        test_pred = [round(_) for _ in test_pred_prob]



# evaluate the performance of methods
def detection_eval(data, model, method_name):
    original_text, sampled_text = load_datasets(data, False)
    if method_name in params.METHOD_PARAMS['supported_pretrained_models'] or method_name == 'roberta_base' or method_name == 'roberta_large' or method_name == 'roberta_sentinel' or model_name == 't5_sentinel':
        classifier_detection(original_text, sampled_text, model, method_name)
    elif method_name in params.METHOD_PARAMS['supported_tools']:
        tool_evaluation(original_text, sampled_text, model, method_name)



