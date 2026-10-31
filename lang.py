import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # reads .env in the current folder

groq_model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
)
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

prompt = ChatPromptTemplate.from_template("Explain {topic} in 2 sentences.")
parser = StrOutputParser()

for name, model in [("Groq", groq_model), ("Gemini", gemini_model)]:
    chain = prompt | model | parser
    print(f"{name}: {chain.invoke({'topic': 'vector databases'})}\n")