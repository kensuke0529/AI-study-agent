import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
import spacy
from wiki import google_search

# === Load environment and models ===
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
nlp = spacy.load("en_core_web_sm")


# === Conversation Memory ===
class ConversationMemory:
    def __init__(self, max_length=5):
        self.memory = []
        self.max_length = max_length

    def add(self, user, ai):
        self.memory.append((user, ai))
        if len(self.memory) > self.max_length:
            self.memory.pop(0)

    def to_message_list(self):
        messages = []
        for user, ai in self.memory:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": ai})
        return messages


# === Routing Strategy ===
def route_query_strategy(query, file_list, faiss_results_exist):
    if faiss_results_exist:
        return "documents"

    router_prompt = f"""
        You are a routing assistant. Based on the question below, decide whether the answer should be retrieved from:

        - documents (if it relates to provided files)
        - web (if it requires current info or external sources), especially user mention 'wiki' or ask you looking for information from the internet. 
        - knowledge (if it's basic or general knowledge)

        Files: {', '.join(file_list)}

       Few-shot examples:
        Q: What's the latest version of JavaScript? 
        A: web

        Q: What's the weather today in London? - you need to look up for weather from the internet. 
        A: web

        Q: What's a variable in programming? 
        A: knowledge

        Q: Explain inheritance in OOP.
        A: knowledge

        Q: {query}
        A:
    """

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "system", "content": router_prompt}]
    )

    decision = response.choices[0].message.content.strip().lower()
    if decision not in ['documents', 'web']:
        decision = 'knowledge'
    return decision


# === Query Handling ===
def answer_query_with_context(query, index, chunks, chunk_doc_names, client, file_list, memory=None, k=3, distance_threshold=1.0):

    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(
        response.data[0].embedding, dtype='float32').reshape(1, -1)

    # FAISS search
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    retrieved_docs = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1 and dist <= distance_threshold:
            retrieved_chunks.append(chunks[idx])
            retrieved_docs.append(chunk_doc_names[idx])

    faiss_has_results = len(retrieved_chunks) > 0

    # Route
    source = route_query_strategy(query, file_list, faiss_has_results)

    # Compose user prompt
    if source == "documents":
        context = "\n\n".join(retrieved_chunks)
        user_prompt = (
            "Based on the following documents, answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    elif source == "knowledge":
        user_prompt = (
            "No relevant documents were found. "
            "Please answer using your general knowledge.\n\n"
            f"Question: {query}"
        )
    elif source == "web":
        try:
            wiki_summary = google_search(query)
            user_prompt = (
                "Based on a summarized Wikipedia reference, answer the question.\n\n"
                f"Wikipedia Summary:\n{wiki_summary}\n\n"
                f"Question: {query}"
            )
        except Exception as e:
            user_prompt = (
                "The web search function encountered an error. "
                "Please answer using your general knowledge.\n\n"
                f"Question: {query}\n\n"
                f"(Error: {str(e)})"
            )
    else:
        raise ValueError("Unknown routing decision.")

    # System instructions
    system_message = (
        "You are an AI study assistant. "
        "Always clarify which information came from context (Documents, Wikipedia, or Knowledge). "
        "If the answer is not present in context, say so before using general knowledge. "
        "Do not hallucinate or invent information. "
        "If unsure, ask clarifying questions."
    )

    # Build chat history
    messages = [{"role": "system", "content": system_message}]
    if memory is not None:
        messages.extend(memory.to_message_list())
    messages.append({"role": "user", "content": user_prompt})

    # Get model response
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    answer = chat_response.choices[0].message.content

    # Update memory
    if memory is not None:
        memory.add(query, answer)

    return {
        "answer": answer,
        "source": source,
        "docs_used": list(set(retrieved_docs)) if retrieved_docs else []
    }
