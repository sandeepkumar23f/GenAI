import os
os.environ["HF_HUB_OFFLINE"] = "0"   # set to "1" after the first download to run fully offline

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",   # small instruct model, about 1 GB
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=150,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False,
    ),
)
model = ChatHuggingFace(llm=llm)

system = ("system", "You are a helpful assistant. Answer briefly.")
history = []

print("-------------- Local Chatbot --------------")
print("----- Please enter 0 to exit -----")

while True:
    prompt = input("You: ").strip()
    if prompt == "0":
        print("Bye!")
        break
    if not prompt:
        continue

    history.append(("user", prompt))
    try:
        response = model.invoke([system] + history[-4:])
    except Exception as e:
        print("Error:", e, "\n")
        history.pop()
        continue

    history.append(("assistant", response.content))
    print("Bot:", response.content, "\n")