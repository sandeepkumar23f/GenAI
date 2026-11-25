import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage


llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task='text-generation',
    pipeline_kwargs=dict(
        max_new_tokens=200,
    )
)

model = ChatHuggingFace(llm=llm)

print("For funny press 1 for Angry press 2 for Sad press 3 for Happy press 4")

choice = input("Enter your mode: ")

if choice == '1':
    mode="You are a Funny AI agent. So reply in funny manner"
elif choice == '2':
    mode="You are a angry AI agent. Reply in angry manner"
elif choice == '3':
    mode="You are sad AI agent. Reply in sad manner"
elif choice == '4':
    mode="You are Happy AI agent. Reply in cheerful manner "
else:
    print("Invalid mode")
    exit()

history=[
    SystemMessage(content=mode)
]

while True:
    prompt = input("You: ")
    
    if prompt == '0':
        print("Chat ended!")
        break
    history.append(HumanMessage(content=prompt))
    
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    print("AI: ",response.content)