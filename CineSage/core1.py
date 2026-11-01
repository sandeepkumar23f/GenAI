import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()
# Creating a MOvie class schema
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1000,
)

prompt = ChatPromptTemplate.from_messages([
    ('system',"""
        Extract Movie information from the paragraph
        {format_instruction} 
    """),
    ("human","{paragraph}")
]
)


para = input("Give your paragraph: ")

final_prompt = prompt.invoke({"paragraph": para,"format_instruction": parser.get_format_instructions()})   # fill in the template
response = model.invoke(final_prompt)               # send it to the model
print(response.content)