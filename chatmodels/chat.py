from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

#using gemini api key
# model = init_chat_model("google_genai:gemini-3.7-flash, max_tokens=200")

# using groq api key
model = init_chat_model(
    "groq:openai/gpt-oss-120b",
    max_tokens=200,   # hard limit on the reply length
)

resp = model.invoke("in 2 sentense What does langchain means in LLMs")
print(resp.content)