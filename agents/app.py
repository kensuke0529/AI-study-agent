import streamlit as st
import os
from pathlib import Path
from qa_agent import *
from vector_store import * 
from document_handling import *


st.set_page_config(page_title="Study Agent", layout="wide")
st.title("📚 Study Assistant Agent")

# 1. Get all topic directories
topic_base = "../documents"
topic_dirs = [d for d in os.listdir(topic_base) if os.path.isdir(os.path.join(topic_base, d))]

# Handle if no folders exist
if not topic_dirs:
    st.warning("No topic folders found inside `documents/`. Please create at least one.")
    st.stop()

# 2. Choose topic
selected_topic = st.selectbox("📂 Choose a topic", topic_dirs)

# 3. Upload files (optional)
uploaded_files = st.file_uploader("📄 Upload new documents (optional)", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_dir = os.path.join(topic_base, selected_topic)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully!")

from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 5. Run preprocessing & embedding
with st.spinner("📥 Checking and embedding new documents..."):
    new_chunks = preprocess_and_save(selected_topic, client)
    if new_chunks:
        st.info(f"Embedded {len(new_chunks)} new chunks.")
   

# 6. Question input
st.subheader("🧾 Ask a question")
question = st.text_input("Type your question about the selected topic")

if question:
    try:
        with st.spinner("💬 Thinking..."):
            index, chunks, chunk_doc_names = build_faiss_index(selected_topic)
            result = answer_query_with_context(question, index, chunks, chunk_doc_names, client)

            st.markdown("### ✅ Answer:")
            st.markdown(result["answer"])

            st.markdown("**Source:**")
            st.markdown(result["source"])

            st.markdown("**Documents used:**")
            for doc in result["docs_used"]:
                st.markdown(f"- {doc}")

    except FileNotFoundError:
        st.error("⚠️ No embeddings found. Please upload at least one document to get started.")

