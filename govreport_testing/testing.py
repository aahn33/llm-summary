import json
import sys
sys.path.append('../')

from pathlib import Path
from langchain.chat_models import ChatOpenAI
from rouge import Rouge

from map_and_refine.map_reduce import MapTextSummarizer
from relevancy_score_tagging.relevancy import RelevancyTagger


def calculate_rouge(title, generated_path, reference):
    with open(generated_path, 'r', encoding='utf-8') as file:
        generated = file.read()

    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)[0]
    score_file = open("%s.txt"%title, "w")
    score_file.write(f"ROUGE-1: Precision = {scores['rouge-1']['p']}, Recall = {scores['rouge-1']['r']}, F1 = {scores['rouge-1']['f']}\n")
    score_file.write(f"ROUGE-2: Precision = {scores['rouge-2']['p']}, Recall = {scores['rouge-2']['r']}, F1 = {scores['rouge-2']['f']}\n")
    score_file.write(f"ROUGE-L: Precision = {scores['rouge-l']['p']}, Recall = {scores['rouge-l']['r']}, F1 = {scores['rouge-l']['f']}\n")
    score_file.close()

# Sample test using a single extracted file
with open(Path('extracted/98-696.json'), encoding='utf-8') as f:
    # Retrieve testing data
    data = json.loads(f.read())

    title = data['title']
    summary = data['summary']
    text = data['full_text']

    # Initialize LLM
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0, openai_api_key="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",
                    model_name=model_name)
    
    # Filter for relevant chunks in text
    tagger = RelevancyTagger(llm, model_name, 0.8, 500)
    tagger.tag(text, title)

    # Get output from full text map_reduce
    summarizer = MapTextSummarizer(llm=llm, model_name=model_name)
    summary, total_tokens_used = summarizer.process(text, 'full')

    print(summary)
    print(f"Total tokens used: {total_tokens_used}")

    # Get output from relevant text map_reduce
    summary, total_tokens_used = summarizer.process(open(Path('relevancy/relevant_chunks.txt')).read(), 'relevant')
    print(summary)

    # Perform rouge test
    calculate_rouge('test_full', 'map/full_final_summary_1.txt', summary)
    calculate_rouge('test_relevant', 'map/relevant_final_summary_1.txt', summary)




