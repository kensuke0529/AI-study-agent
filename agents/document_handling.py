import os
import json
import hashlib
from pathlib import Path
import numpy as np
from pypdf import PdfReader
import spacy 

nlp = spacy.load('en_core_web_sm')


def list_topic_files(topic_name):
    """Return list of .pdf and .txt filenames in topic folder."""
    topic_path = Path(f'../documents/{topic_name}')
    return [
        f.name for f in topic_path.iterdir()
        if f.is_file() and f.suffix.lower() in [".pdf", ".txt"]
    ]

    return [f.name for f in topic_path.iterdir() if f.is_file() and f.suffix.lower() in [".pdf", ".txt"]]

def extract_text(file_path: Path) -> str:
    """Extract text from a txt or pdf file."""
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    else:
        raise ValueError("Only .txt and .pdf files supported")

# def chunk_text(text, chunk_size=500, overlap=50):
#     """Split text into overlapping chunks."""
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks

def chunk_by_sentence(text, max_sentences=4, overlap=2):
    doc = nlp(text)
    sentences = list(doc.sents)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sent.text for sent in sentences[i:i+max_sentences])
        if chunk:
            chunks.append(chunk)
        if i + max_sentences >= len(sentences):
            break
        i += max_sentences - overlap  
    return chunks

def file_hash(file_path: Path):
    """Calculate SHA256 hash of a file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# ----- Main preprocessing and saving function -----

def preprocess_and_save(topic_name, client):
    topic_path = Path(f'../documents/{topic_name}')
    topic_path.mkdir(parents=True, exist_ok=True)

    metadata_file = topic_path / "metadata.json"
    embeddings_file = topic_path / "chunk_embeddings.npy"

    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        prev_hashes = metadata.get("file_hashes", {})
        old_chunks = metadata.get("chunks", [])
        old_doc_names = metadata.get("chunk_doc_names", [])
    else:
        prev_hashes = {}
        old_chunks = []
        old_doc_names = []

    all_chunks = []
    all_chunk_doc_names = []
    new_hashes = {}

    for filename in list_topic_files(topic_name):
        file_path = topic_path / filename
        current_hash = file_hash(file_path)
        new_hashes[filename] = current_hash

        if prev_hashes.get(filename) == current_hash:
            print(f"Skipping unchanged file: {filename}")
            continue

        print(f"Processing new/updated file: {filename}")
        text = extract_text(file_path)
        chunks = chunk_by_sentence(text)
        all_chunks.extend(chunks)
        all_chunk_doc_names.extend([filename] * len(chunks))

    if not all_chunks:
        print("No new or updated chunks to embed.")
        return

    print(f"Embedding {len(all_chunks)} new chunks...")
    response = client.embeddings.create(
        input=all_chunks,
        model="text-embedding-3-small"
    )
    new_embeddings = [np.array(item.embedding, dtype="float32") for item in response.data]
    new_embeddings_np = np.stack(new_embeddings)

    if old_chunks and old_doc_names and embeddings_file.exists():
        old_embeddings = np.load(embeddings_file)
        all_chunks = old_chunks + all_chunks
        all_chunk_doc_names = old_doc_names + all_chunk_doc_names
        embeddings_np = np.vstack([old_embeddings, new_embeddings_np])
    else:
        embeddings_np = new_embeddings_np

    np.save(embeddings_file, embeddings_np)

    metadata = {
        "chunks": all_chunks,
        "chunk_doc_names": all_chunk_doc_names,
        "file_hashes": new_hashes,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print("Saved embeddings and metadata.")
