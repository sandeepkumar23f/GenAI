import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 1. Embed one sentence
vec = embeddings.embed_query("Python is a programming language")
print("Dimensions:", len(vec))
print("First 5 numbers:", vec[:5])

# 2. Tiny semantic search over a few documents
docs = [
    "Python is a popular programming language",
    "Vector databases store embeddings for similarity search",
    "Pizza is best with extra cheese",
    "Groq runs large language models very fast",
]
doc_vectors = np.array(embeddings.embed_documents(docs))

question = "Which language is good for coding?"
q_vector = np.array(embeddings.embed_query(question))

# cosine similarity between the question and each document
scores = doc_vectors @ q_vector / (
    np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q_vector)
)

print("\nQuestion:", question)
for doc, score in sorted(zip(docs, scores), key=lambda x: -x[1]):
    print(f"{score:.3f}  {doc}")