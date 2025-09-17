import os
import json
import faiss
import numpy as np

from google import genai
from google.genai.types import EmbedContentConfig

# ---------------------------
# Setup Gemini client
# ---------------------------
your_api_key = "AIzaSyB_f7xSkHydvZpZLBshpCpjk9Z9UexbmhQ"

genai_client = genai.Client(api_key=your_api_key)

class GeminiEmbeddingBackend:
    def __init__(self, model: str = "gemini-embedding-001"):
        self.model = model

    def embed_texts(self, texts):
        embeddings = []
        batch_size = 100  # Gemini limit
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            config = EmbedContentConfig()
            response = genai_client.models.embed_content(
                model=self.model,
                contents=batch,
                config=config
            )
            for emb in response.embeddings:
                embeddings.append(np.array(emb.values, dtype=np.float32))
        return np.vstack(embeddings)

# ---------------------------
# Build FAISS index
# ---------------------------
def build_faiss_from_json(json_path, index_path, meta_path, backend):
    # Load ISL dictionary
    with open(json_path, "r", encoding="utf-8") as f:
        isl_dict = json.load(f)

    # Prepare texts (each entry: "word: X, isl: Y")
    texts = [f"word: {w}, isl: {t}" for w, t in isl_dict.items()]
    metadata = [{"word": w, "isl": t} for w, t in isl_dict.items()]

    # Embed all texts
    print("Embedding dictionary entries with Gemini...")
    embeddings = backend.embed_texts(texts)
    faiss.normalize_L2(embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized = cosine similarity
    index.add(embeddings)

    # Save FAISS + metadata
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ FAISS DB created with {index.ntotal} entries.")
    return index, metadata


if __name__ == "__main__":
    JSON_PATH = "isl_dict.json"          # your dictionary file
    INDEX_PATH = "isl_faiss.index"
    META_PATH = "isl_faiss_meta.json"

    backend = GeminiEmbeddingBackend()
    build_faiss_from_json(JSON_PATH, INDEX_PATH, META_PATH, backend)
