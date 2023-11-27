import json
import os
import pandas as pd
import sys
sys.path.append('../')

from pathlib import Path
from langchain.chat_models import ChatOpenAI
from rouge import Rouge

from map_and_refine.map_reduce import MapTextSummarizer


def calculate_rouge(summary, reference):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)[0]
    return scores


def save_results(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    flattened = [pd.DataFrame(pd.json_normalize(results))]
    df = pd.concat(flattened, ignore_index=True)
    df.to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path), index=False)


# Initialize LLM
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0, openai_api_key="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",
                model_name=model_name, request_timeout=60)

# Open extracted files
files = list(Path('extracted').glob("*"))
for file in files:
    with open(file, encoding='utf-8') as f:
        # Retrieve testing data
        data = json.loads(f.read())
        title = data['title']
        ground_truth = data['summary']
        full_text = data['full_text']

        print(f"Processing file {file}")

        # Get output from full text map_reduce
        summarizer = MapTextSummarizer(llm=llm, model_name=model_name)
        summary, total_tokens_used = summarizer.process(full_text)

        # Perform rouge test
        scores = calculate_rouge(summary, ground_truth)
        scores['tokens_used'] = total_tokens_used
        scores['test_file'] = file

        # Append results to csv file
        save_results(scores, Path('results/map_reduce/results.csv'))

        # Also save the summary just in case
        basename, _ = os.path.splitext(os.path.basename(file))
        with open(Path(f'results/map_reduce/summary_{basename}.txt'), 'w') as out:
            out.write(summary)

        print(f"Results for {file} saved")
        print()



