import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# PART 1: LOGIC 

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
    ("system", """
        Extract Movie information from the paragraph
        {format_instruction}
    """),
    ("human", "{paragraph}")
])


def extract(para):
    """Your original steps: fill the prompt, call the model, return the text."""
    final_prompt = prompt.invoke({
        "paragraph": para,
        "format_instruction": parser.get_format_instructions(),
    })
    response = model.invoke(final_prompt)
    return response.content


# =====================================================
# PART 2: UI (Streamlit only, no logic here)
# =====================================================
st.set_page_config(page_title="CineSage", page_icon="🎬")
st.title("🎬 CineSage")
st.caption("Paste a paragraph about a movie to extract its information")

para = st.text_area("Give your paragraph", height=200)   # replaces input(...)

if st.button("Extract"):
    if not para.strip():
        st.warning("Please enter a paragraph first.")
    else:
        try:
            with st.spinner("Extracting..."):
                result = extract(para)                    # LOGIC call
            st.subheader("Result")
            st.code(result, language="json")              # replaces print(response.content)
        except Exception as e:
            st.error(f"Error: {e}")