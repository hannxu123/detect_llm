
import transformers




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