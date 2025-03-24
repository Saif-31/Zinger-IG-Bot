import streamlit as st
from app import create_bot, KNOWLEDGE_BASE, AgentState, process_message
from langchain_core.messages import SystemMessage, HumanMessage

# Add sidebar with New Chat button
with st.sidebar:
    if st.button("New Chat"):
        # Clear chat history and reset bot state
        st.session_state.messages = []
        st.session_state.state = {
            "messages": [SystemMessage(content="")],
            "knowledge_base": KNOWLEDGE_BASE
        }
        st.rerun()
        
st.title("Zinger Interior Design Bot")
st.markdown("###### Welcome to the Zinger Assistant Chatbot! Type your message below to start a conversation.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.bot = create_bot(KNOWLEDGE_BASE)
    st.session_state.state = {
        "messages": [SystemMessage(content="")],
        "knowledge_base": KNOWLEDGE_BASE
    }

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add user message to bot state
    st.session_state.state["messages"].append(HumanMessage(content=prompt))
    
    # Get bot response using process_message directly
    result = process_message(st.session_state.state)
    ai_messages = result.get("messages", [])
    
    # Add bot response to chat history
    for msg in ai_messages:
        st.session_state.messages.append({"role": "assistant", "content": msg.content})
        st.session_state.state["messages"].append(msg)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
