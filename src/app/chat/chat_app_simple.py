import sys
import os
import streamlit as st

# fmt: off
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from IPython.display import display, Image

# fmt: on
from app.chat.state_machine import graph

# from app.chat.supervisor_graph import graph


def stream_graph_updates(user_input: str, config: dict):
    respuesta = ""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]}, config
    ):
        for value in event.values():
            partial = value["messages"][-1].content
            respuesta += partial
            yield respuesta


config = {"configurable": {"thread_id": "1"}}

st.title("Chat LangGraph Demo")

menu = st.sidebar.selectbox("Selecciona una ventana", ["Chatbot", "Grafo Agente"])

if menu == "Chatbot":
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

elif menu == "Grafo Agente":
    st.subheader("Visualización del Grafo del Agente")
    st.info(
        "Aquí puedes mostrar la visualización o información relevante del grafo del agente."
    )
    # Aquí puedes agregar el código para visualizar el grafo o mostrar información adicional.

    st.image(
        graph.get_graph().draw_mermaid_png(),
        caption="Grafo del Agente",
        use_column_width=True,
    )

    # Image(graph.get_graph().draw_mermaid_png())
