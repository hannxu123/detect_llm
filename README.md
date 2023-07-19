# Detect LLM generated texts

Our design to build new LLM-generated datasets. It includes 4 major tasks and each task has 3 topics. Each topics have 2,000 ChatGPT and 2,000 human. (1 Naive version, 1 human-imitation version?)

1. News generation. 

2. Product (Amazon) / Movie (IMDb) Review.

3. Creative writing: ivypanda essays, [dataset link](https://huggingface.co/datasets/qwedsacf/ivypanda-essays/)) ASAP dataset (only 8 prompts)

4. Question Answer from HC-3. 

### *Part 1. Useful links* 

Our overleaf link: https://www.overleaf.com/8822696486dnqvwmwfkfbn

News dataset is here: https://drive.google.com/file/d/1yidALJz2C7DBS_vwOpD1DT2TrxJFXQEt/view?usp=sharing 

Download the [IMDb Movie Dataset ](https://github.com/sahildit/IMDB-Movies-Extensive-Dataset-Analysis/blob/master/data1/IMDb%20movies.csv), to find movie infomration. 

Awesome ChatGPT Prompts: https://github.com/f/awesome-chatgpt-prompts  

You can make API requests one by one. However, to make it faster, you can try the api_request_parallel_processor.py file to make the requests at scale. But you need to follow the instructions in the api_request_parallel_processor.py file. **Remember to use "max_tokens" to limit the length of each output to avoid high cost.**
This requires to make a jsonl file. To easily write a jsonl file, you can follow the example in example.py. Then you can query the api_request_parallel_processor.py:
```
python api_request_parallel_processor.py --requests_filepath data.jsonl --request_url https://api.openai.com/v1/chat/completions
```


