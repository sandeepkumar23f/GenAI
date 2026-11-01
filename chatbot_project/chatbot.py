import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage

load_dotenv()  
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
    max_tokens=500,
)

history = [
    SystemMessage(content="You are a funny AI agent")
]
print("--------------Welcome to Chatbot--------------")
print("-----Please enter 0 to exit------")

while True:
    prompt = input("You: ")
    history.append(HumanMessage(content=prompt))

    if prompt == '0':
        break
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    print(response.content)
print(history)