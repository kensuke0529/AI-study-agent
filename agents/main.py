import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment and initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Your internal modules
from qa_agent_gate import answer_query_with_context
from document_handling import preprocess_and_save, load_chunks_from_cache
from agents.vector_store import load_faiss_index

if __name__ == "__main__":
    # Step 1: Choose topic
    topic = input("Choose your topic: ")

    print("\nStep 1: Preprocess and embed documents (if new or changed)...")
    preprocess_and_save(topic, client)  # This should create embeddings and save chunks

    print("\nStep 2: Load FAISS index and associated data...")
    index = load_faiss_index(topic)
    chunks, chunk_doc_names = load_chunks_from_cache(topic)  # Load saved chunk texts + doc names

    file_list = list(set(chunk_doc_names))  # For routing decisions

    print("\nStep 3: Ask your question.")
    user_question = input("Ask me anything: ")

    result = answer_query_with_context(
        query=user_question,
        index=index,
        chunks=chunks,
        chunk_doc_names=chunk_doc_names,
        client=client,
        file_list=file_list
    )

    print("\n📌 Answer:")
    print(result["answer"])
    print("\n📄 Source:", result["source"])
    print("📚 Documents used:", result["docs_used"])
