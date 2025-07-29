import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from qa_agent_gate import answer_query_with_context, ConversationMemory
from document_handling import preprocess_and_save
from vector_store import build_faiss_index

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.set_page_config(page_title="Study Agent", layout="wide")
st.title("Study Assistant Agent")

topic_base = "../documents"
topic_dirs = [d for d in os.listdir(topic_base) if os.path.isdir(os.path.join(topic_base, d))]
if not topic_dirs:
    st.warning("No topic folders found in documents/.")
    st.stop()

selected_topic = st.selectbox("Choose a topic", topic_dirs)
uploaded = st.file_uploader("Upload new documents", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded:
    for uf in uploaded:
        save_dir = os.path.join(topic_base, selected_topic)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
    st.success("Uploaded successfully!")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

with st.spinner("Checking and embedding..."):
    new_chunks = preprocess_and_save(selected_topic, client)
    if new_chunks:
        st.info(f"Embedded {len(new_chunks)} new chunks.")

question = st.text_input("Ask a question")

if question:
    try:
        index, chunks, chunk_names = build_faiss_index(selected_topic)
        file_list = list(set(chunk_names))
        with st.spinner("Thinking..."):
            result = answer_query_with_context(
                query=question,
                index=index,
                chunks=chunks,
                chunk_doc_names=chunk_names,
                client=client,
                file_list=file_list,
                memory=st.session_state.memory
            )
        st.markdown("### Answer:")
        st.markdown(result["answer"])
        st.markdown("**Source:**")
        st.markdown(result["source"])
        st.markdown("**Documents used:**")
        for d in result["docs_used"]:
            st.markdown(f"- {d}")

    except Exception as e:
        st.error(f"Error: {e}")
