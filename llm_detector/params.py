

DATASET_PARAMS = {
    'supported_public_datasets': ['open-web-text', 'open-gpt-text', 'hc3'],
    'supported_model_generated_datasets': ['xsum', 'squad', 'writing', 'pubmed', 'wmt16_en', 'wmt16_de'],
}


METHOD_PARAMS = {
    'supported_pretrained_models': ['gpt2_detector_roberta_base', 'gpt2_detector_roberta_large', 'open_gpt_text_roberta_sentinel', 'open_gpt_text_t5_sentinel'],
    'supported_methods': ['roberta_base', 'roberta_large', 'roberta_sentinel', 't5_sentinel', 'detect_gpt', 'gpt_pat', 'mpg_roberta_base', 'roberta_large', 'black_box_watermark'],
    'supported_tools': ['gpt_zero'],
}


EXP_TEMP_PARAMS = {
    'prompt': 'all1',
    'padding_max_length': 256,
    'dataset_name': 'ELI5',
    'batch_size': 32
}


OTHERS_PARAMS = {
    'padding_max_length': 256,
    'test_batch_size': 32,
}


ROBERTA_PARAMS = {
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'weight_decay': 0,
    'optimizer': 'Adam',
}


ROBERTA_SENTINEL_PARAMS = {
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-3,
    'optimizer': 'AdamW',
    'accumulaiton_steps': 8,
}


T5_SENTINEL_PARAMS = {
    'num_epochs': 5,
    'batch_size': 32,
    'learning_rate': 5e-4,
    'weight_decay': 1e-3,
    'optimizer': 'AdamW',
    'accumulaiton_steps': 8,
}


MPU_ROBERTA_PARAMS = {
    'num_epochs': 5,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'weight_decay': 0,
    'prior': 0.2, 
    'pu_type': 'dual_softmax_dyn_dtrun',
    'lamb': 0.4, 
    'len_thres': 55,
}


GPT_PAT_PARAMS = {
    
}

