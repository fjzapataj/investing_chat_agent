import sys
import os
import streamlit as st

# fmt: off
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) #'../..'

# fmt: on
# from app.chat.state_machine import graph
from app.chat.supervisor_graph import graph


def stream_graph_updates(user_input: str, config: dict):
    respuesta = ""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, config
    ):
        for value in event.values():
            # Debug: print the intermediate value
            print("DEBUG value:", value)
            # Print the messages key if it exists
            if "messages" in value:
                print("DEBUG value['messages']:", value["messages"])
                # Check if messages is a list and not empty
                if isinstance(value["messages"], list) and value["messages"]:
                    # Print the last message
                    print("DEBUG last message:", value["messages"][-1])
                    # Check if the last message has 'content' as attribute or key
                    last_msg = value["messages"][-1]
                    if isinstance(last_msg, dict) and "content" in last_msg:
                        partial = last_msg["content"]
                    elif hasattr(last_msg, "content"):
                        partial = last_msg.content
                    else:
                        print("DEBUG: last_msg has no 'content'")
                        partial = ""
                else:
                    print("DEBUG: value['messages'] is not a non-empty list")
                    partial = ""
            else:
                print("DEBUG: 'messages' not in value")
                partial = ""
            respuesta += partial
            yield respuesta


config = {"configurable": {"thread_id": "1"}}

st.title("Chat LangGraph Demo")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**Usuario:** {msg['content']}")
    else:
        st.markdown(f"**Asistente:** {msg['content']}")

with st.form("form_mensaje", clear_on_submit=True):
    user_input = st.text_input("Escribe tu mensaje:")
    enviar = st.form_submit_button("Enviar")

if enviar and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    respuesta = ""
    message_placeholder = st.empty()
    for partial in stream_graph_updates(user_input, config):
        message_placeholder.markdown(f"**Asistente:** {partial}")
        respuesta = partial
    st.session_state.messages.append({"role": "assistant", "content": respuesta})
