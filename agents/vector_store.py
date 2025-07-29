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

def build_faiss_index(topic_name):
    topic_path = Path(f'../documents/{topic_name}/metadata')
    embeddings_file = topic_path / "chunk_embeddings.npy"
    metadata_file = topic_path / "metadata.json"

    if not (embeddings_file.exists() and metadata_file.exists()):
        raise FileNotFoundError("Embeddings or metadata file missing. Run preprocess_and_save() first.")

    embeddings = np.load(embeddings_file)
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    chunks = metadata.get("chunks", [])
    chunk_doc_names = metadata.get("chunk_doc_names", [])

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index loaded with {index.ntotal} vectors.")

    return index, chunks, chunk_doc_names