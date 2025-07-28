import os
import json
import hashlib
from pathlib import Path
import numpy as np
import faiss
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import spacy 

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
nlp = spacy.load('en_core_web_sm')

# ----- Query function -----

def answer_query_with_context(query, index, chunks, chunk_doc_names, client, k=3, distance_threshold=1.0):
    
    
    # Embed query
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)

    # Search FAISS index
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    retrieved_docs = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1 and dist <= distance_threshold:
            retrieved_chunks.append(chunks[idx])
            retrieved_docs.append(chunk_doc_names[idx])

# ===================================================
### Prompt Engineering

    if retrieved_chunks:
        context = "\n\n".join(retrieved_chunks)
        user_prompt = (
            "Based on the following documents, answer the question below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        source = "documents"
    else:
        user_prompt = (
            " No relevant documents were found for this question. "
            "Please answer using your general knowledge.\n\n"
            f"Question: {query}"
        )
        source = "general_knowledge"

    system_message = (
        "You are an AI study assistant. "
        "Always base your answer on the provided context from the reference documents. "
        "If the answer is not present, say so and only then use your general knowledge, "
        "but clarify which information came from context. "
        "If unsure, ask clarifying questions. Do not hallucinate or invent information."
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = chat_response.choices[0].message.content
    docs_used = list(set(retrieved_docs)) if retrieved_docs else []

    return {
        "answer": answer,
        "source": source,
        "docs_used": docs_used
    }

# ===================================================

