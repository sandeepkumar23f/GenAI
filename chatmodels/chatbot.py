
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
    max_tokens=100,
)

history = []
print("--------------Welcome to Chatbot--------------")
print("-----Please enter 0 to exit------")

while True:
    prompt = input("You: ")
    history.append(prompt)
    if prompt == '0':
        break
    response = model.invoke(history)
    history.append(response)
    print(response.content)