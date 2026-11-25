from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

import numpy as np
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

chunks = [
    "Our refund policy allows returns within 30 days of purchase.",
    "The office is open Monday to Friday, 9 AM to 6 PM.",
    "To reset your password, click 'Forgot Password' on the login page.",
    "We ship to over 50 countries, and delivery takes 5-7 business days.",
    "Artificial intelligence is transforming healthcare and education.",
]

chunk_vectors = np.array(embedding.embed_documents(chunks))

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def search(query, top_k=2):
    query_vector = np.array(embedding.embed_query(query))

    scores = [cosine_similarity(query_vector,v) for v in chunk_vectors]

    ranked = sorted(zip(scores, chunks), reverse=True)
    return ranked[:top_k]
for q in ["How do I get my money back?", "When can I visit the office?", "Is AI good for society?"]:
    print(f"\nQuery: {q}")
    for score, text in search(q):
        print(f"  {score:.3f}  {text}")