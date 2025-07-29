import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
import spacy

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
        - web (if it requires current info or external sources)
        - knowledge (if it's basic or general knowledge)

        Files: {', '.join(file_list)}

       Few-shot examples:
        Q: What's the latest version of JavaScript?
        A: web

        Q: What's the weather today in London?
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
    if decision not in ['documents', 'web', 'knowledge']:
        decision = 'knowledge'
    return decision


# === Query Handling ===
def answer_query_with_context(
    query,
    index,
    chunks,
    chunk_doc_names,
    client,
    file_list,
    memory=None,  # ✅ properly passed as a parameter
    k=3,
    distance_threshold=1.0
):
    # Step 1: Embed query
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = np.array(response.data[0].embedding, dtype='float32').reshape(1, -1)

    # Step 2: FAISS search
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    retrieved_docs = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1 and dist <= distance_threshold:
            retrieved_chunks.append(chunks[idx])
            retrieved_docs.append(chunk_doc_names[idx])

    faiss_has_results = len(retrieved_chunks) > 0

    # Step 3: Route
    source = route_query_strategy(query, file_list, faiss_has_results)

    # Step 4: Compose user prompt
    if source == "documents":
        context = "\n\n".join(retrieved_chunks)
        user_prompt = (
            "Based on the following documents, answer the question below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
    elif source == "knowledge":
        user_prompt = (
            "No relevant documents were found for this question. "
            "Please answer using your general knowledge.\n\n"
            f"Question: {query}"
        )
    elif source == "web":
        user_prompt = "currently working on internet search function"
    else:
        raise ValueError("Unknown routing decision.")

    # Step 5: Compose system message
    system_message = (
        "You are an AI study assistant. "
        "Always base your answer on the provided context from the reference documents. "
        "If the answer is not present, say so and only then use your general knowledge. "
        "Clarify which information came from context. "
        "If unsure, ask clarifying questions. Do not hallucinate or invent information. "
        "When the prompt is 'currently working on internet search function' then just respond with that same message."
    )

    # Step 6: Build chat history
    messages = [{"role": "system", "content": system_message}]
    if memory is not None:
        messages.extend(memory.to_message_list())
    messages.append({"role": "user", "content": user_prompt})

    # Step 7: Get model response
    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    answer = chat_response.choices[0].message.content

    # Step 8: Update memory
    if memory is not None:
        memory.add(query, answer)

    return {
        "answer": answer,
        "source": source,
        "docs_used": list(set(retrieved_docs)) if retrieved_docs else []
    }
