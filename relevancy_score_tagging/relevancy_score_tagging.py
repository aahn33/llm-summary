import os
from time import sleep
from math import ceil
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter


class RelevancyTagger:
    def __init__(self, llm, model_name, threshold, chunk_size):
        self.llm = llm
        self.model_name = model_name
        self.total_tokens = 0
        self.threshold = threshold
        self.chunk_size = chunk_size

    def split_text(self, file):
        f = open(file, encoding='utf-8')
        text = f.read()

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=0
        )

        texts = text_splitter.split_text(text)
        print(f'Document was split into {len(texts)} chunks')
        return texts

    def prompt_llm(self, texts):
        sys_message = SystemMessage(content=(
                "You are a knowledgeable assistant that takes in a chunk of a document and outputs a score from 0-100 " +
                "depending on how relevant it is to the entire document. You only output the numerical score and nothing else."
        ))

        results = []
        total_tokens = 0
        for i, chunk in enumerate(texts[20:23]):  # Limit chunks for now just to make sure it works
            with get_openai_callback() as cb:
                print(f'Processing chunk {i}')
                results.append(self.model_name([sys_message, HumanMessage(content=chunk)]))
                total_tokens += cb.total_tokens
                sleep(0.1)  # Rate limits

        print(results)
        return results

    def extract_relevant_chunks(self, texts, results):
        scores = [(int(msg.content), idx) for idx, msg in enumerate(results)]
        scores.sort(reverse=True, key=lambda x: (x[0], -x[1]))  # If scores are tied, prefer earlier chunk
        num_chunks = ceil(len(scores) * THRESHOLD)
        print(f"Using a threshold of {THRESHOLD}, {num_chunks} chunks were selected out of {len(scores)}")

        scores = scores[:num_chunks]
        relevant_chunks = [texts[idx] for _, idx in scores]
        print(f"Top {num_chunks} highest scores: {' '.join(str(score) for score, idx in scores)}")
        print(f"Tokens used: {self.total_tokens}")
        print(relevant_chunks)
        return relevant_chunks

    def tag(self, file, iteration):
        texts = tagger.split_text(file)
        results = tagger.prompt_llm(texts)
        relevant_chunks = tagger.extract_relevant_chunks(texts, results)
        self.save_to_file(relevant_chunks, "cur_relevant_chunks", iteration)

    def save_to_file(self, relevant_chunks, name, iteration):
        directory = "relevancy"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"{name}_{iteration}.txt"
        with open(os.path.join(directory, filename), 'w', encoding='utf-8') as file:
            file.write(relevant_chunks)


# Usage of the Relevancy Tagger Class
CHUNK_SIZE = 1000
THRESHOLD = 0.5
iteration = 1 # dummy value for now, but can implement repeating logic if needed
file_path = "Gatsby.txt"
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0, openai_api_key="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",
                 model_name=model_name)
tagger = RelevancyTagger(llm, model_name, THRESHOLD, CHUNK_SIZE)
tagger.tag(file_path, iteration)
