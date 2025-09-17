import json
import pickle
import faiss
import numpy as np
import spacy
import ollama

# ----------------------------
# Load KB + FAISS Index
# ----------------------------
with open("isl_dict.json", "r") as f:
    kb_dict = json.load(f)

with open("index_data/texts.pkl", "rb") as f:
    kb_words = pickle.load(f)

index = faiss.read_index("index_data/vocab.index")
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# Pronoun mapping
# ----------------------------
PRONOUN_MAP = {
    "i": "ME",
    "me": "ME",
    "you": "YOU",
    "he": "HE",
    "she": "SHE",
    "it": "IT",
    "we": "WE",
    "us": "US",
    "they": "THEY",
    "them": "THEY"
}

# ----------------------------
# Embedding helper (Ollama + mxbai)
# ----------------------------
embedding_cache = {}

def get_embedding(word: str):
    if word in embedding_cache:
        return embedding_cache[word]
    response = ollama.embeddings(
        model="mxbai-embed-large:latest",
        prompt=word
    )
    emb = np.array(response["embedding"], dtype=np.float32)
    embedding_cache[word] = emb
    return emb

def find_closest_kb_word(word, threshold=0.6):
    emb = np.array([get_embedding(word)])
    faiss.normalize_L2(emb)
    D, I = index.search(emb, 1)
    if D[0][0] > threshold:
        return None
    return kb_words[I[0][0]]

# ----------------------------
# Full pipeline: LLM directly generates KB sentence
# ----------------------------
def kb_constrained_sentence(input_sentence, threshold=0.6):
    doc = nlp(input_sentence)
    processed_tokens = []

    for token in doc:
        word = token.text.lower()

        # Pronoun handling
        if word in PRONOUN_MAP:
            processed_tokens.append(PRONOUN_MAP[word])
            continue

        # Content words → map via FAISS
        if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
            kb_word = find_closest_kb_word(word, threshold=threshold)
            if kb_word:
                processed_tokens.append(kb_word)
            else:
                processed_tokens.append(word)
        else:
            processed_tokens.append(word)

    # Join processed tokens into a raw KB-style sentence
    kb_raw = " ".join(processed_tokens)

    # Prepare KB list for rules
    kb_list = ", ".join(kb_words)

    # Full rules with examples
    prompt = f"""
You are an assistant that converts English sentences into ISL gloss.

Rules:
1. Output MUST be in gloss format (short sequence of KB words).
2. Do NOT output explanations, full sentences, or commentary.
3. Only use KB words or mapped pronouns (I → ME, you → YOU, etc.).
4. Keep grammar minimal and in ISL word order.
5. Output ONLY the gloss, nothing else.

Examples:
English: "How are you?"
Gloss: "YOU HOW FEEL"

English: "What are you doing?"
Gloss: "YOU WHAT DO"

Preprocessed input (already KB-mapped): "{kb_raw}"
KB words: {kb_list}
"""

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": "You are an expert at creating KB-constrained sentences for ISL gloss."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"].strip()



# ----------------------------
# Demo
# ----------------------------
if __name__ == "__main__":
    sentences = [
        "In the park, a dog chases a ball.",
        "I can't believe you did that!",
        "Go Away!"
    ]
    for s in sentences:
        kb_sentence = kb_constrained_sentence(s)
        print("Input:", s)
        print("KB-constrained sentence:", kb_sentence)
        print()