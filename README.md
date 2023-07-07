# Detect LLM generated texts

Our design to build new LLM-generated datasets. It includes 4 major tasks and each task has 3 topics. Each topics have 2,000 ChatGPT and 2,000 human. 

1. Society / Politics / Sports / Entertainment News generation:
Base (human-written) dataset: Xsum, a dataset contains human-written news articles accompanied with a one-sentence summary.
using the prompt = "Write an article following summary: Si".

2. Product (Amazon) / Movie (IMDb) / Scientific Paper (XXX) review:

3. Use ChatGPT to create stories 
GRE / SAT Essay, HS standardized testing essays are difficult to find; not released by the organizations, Datasets\\
ivypanda essays, [dataset link](https://huggingface.co/datasets/qwedsacf/ivypanda-essays/)) ASAP dataset (only 8 prompts)

4. Question Answer: 
Science, medical, finance, XXX	
Datasets
HC3
alpaca-gpt4
ELI5

### *Part 1. IMDb Movie & Amazon Product Reviews* 

Download the [IMDb Movie Dataset ](https://github.com/sahildit/IMDB-Movies-Extensive-Dataset-Analysis/blob/master/data1/IMDb%20movies.csv), to find movie infomration. 
Use ChatGPT to generate reviews given the movie names. For example, you can try “Write a negative review about the movie "\<MovieName\>", which is directed by \<DirectorName\>. It talks about "\<Description\>". Just give me the review text.” Collect 500 positive reviews and negative reviews and each review is for 1 movie. Human written reviews can be collected from https://huggingface.co/datasets/imdb. **Remember to use "max_tokens=300" to limit the length of each output to avoid high cost.**

Awesome ChatGPT Prompts: https://github.com/f/awesome-chatgpt-prompts  


*Write some comment about the movie Interstellar like in a conversation. Following the style of this review: "Lifetime did it again. Can we say stupid? I couldn't wait for it to end. The plot was senseless. The acting was terrible! Especially by the teenagers. The story has been played a thousand times! Are we just desperate to give actors a job? The previews were attractive and I was really looking for a good thriller.Once in awhile lifetime comes up with a good movie, this isn't one of them. Unless one has nothing else to do I would avoid this one at all cost. This was a waste of two hours of my life. Can I get them back? "*


```python
def run_gpt(prompt):
    completions = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        max_tokens=300,
        top_p = 0.9,
        messages=[{'role': 'user', 'content': prompt}]
    )
​
    message = completions['choices'][0]['message']['content']
    return message.strip()
```

You can make API requests one by one. However, to make it faster, you can try the api_request_parallel_processor.py file to make the requests at scale. But you need to follow the instructions in the api_request_parallel_processor.py file. **Remember to use "max_tokens" to limit the length of each output to avoid high cost.**

This requires to make a jsonl file. To easily write a jsonl file, you can follow the example in example.py. Then you can query the api_request_parallel_processor.py:
```
python api_request_parallel_processor.py --requests_filepath data.jsonl --request_url https://api.openai.com/v1/chat/completions
```
Use the similar strategy to for Amazon product review. https://nijianmo.github.io/amazon/index.html 
