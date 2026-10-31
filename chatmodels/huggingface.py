import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
load_dotenv()
print("Token loaded:", bool(os.getenv("HF_TOKEN")))
llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
)

model = ChatHuggingFace(llm = llm)

response = model.invoke("Explain RAG in 200 sentence !")

print(response.content)