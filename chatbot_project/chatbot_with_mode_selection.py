import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage,HumanMessage
load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
    max_tokens=100,
)

print("------ Hello Welcome -------- ")

print("Press 1 for funny, 2 for Sad, 3 for Techy, 4 for Simple Explanation")
print(" Press 0 to exit ")

coice = int(input("Enter your choice: "))

if coice == 1:
    mode = "You are a very funny AI agent. Reply in Funny way for every questions."
elif coice == 2:
    mode = "You are a Angry AI agent. Reply in Sad way for every qeustions."
elif coice == 3:
    mode = "You are a Techy AI agent. Reply in Technical like you are an expert of that field for every qeustions."
else:
    mode = "You are a AI agent. who explains every things in very easy ways."
history=[
    SystemMessage(content=mode)
]

while True:
    prompt = input("You: ")
    history.append(HumanMessage(content=prompt))
    if prompt == '0':
        break

    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    print(response.content)