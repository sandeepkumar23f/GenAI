import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.5,
    max_tokens=500,
)

# UI: page title (new)
st.title("Welcome to Chatbot")
st.caption("Please enter 0 to exit")

# Streamlit reruns the script on every message, so history is kept in session_state
if "history" not in st.session_state:
    st.session_state.history = [SystemMessage(content="You are a Sad AI agent. Reply every answer in sad")]
    st.session_state.ended = False

history = st.session_state.history

# UI: show earlier messages (skips the system message)
for msg in history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# UI: input box (replaces input("You: "))
prompt = st.chat_input("You: ", disabled=st.session_state.ended)

if prompt:
    history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    if prompt == '0':
        st.session_state.ended = True
        st.write("Bye!")
        st.write(history)   # same as your final print(history)
        st.stop()

    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    with st.chat_message("assistant"):
        st.write(response.content)