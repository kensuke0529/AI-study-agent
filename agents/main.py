import os 
from dotenv import load_dotenv
from openai import OpenAI
from qa_agent import * 
from document_handling import preprocess_and_save
from agents.vector_store import *

if __name__ == "__main__":
    topic = input('Choose your topic:  ')
    print("Step 1: Preprocess and embed documents (if changed)")
    preprocess_and_save(topic, client)

    print("\nStep 2: Build/load FAISS index")
    index, chunks, chunk_doc_names = build_faiss_index(topic)

    print("\nStep 3: Ask a question")
    user_question = input('Ask me anything: ')
    result = answer_query_with_context(user_question, index, chunks, chunk_doc_names, client)

    print("\nAnswer:")
    print(result["answer"])
    print("\nSource:", result["source"])
    print("Documents used:", result["docs_used"])