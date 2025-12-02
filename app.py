import streamlit as st
from chatbot import rag  # prilagodi naziv fajla ako nije chatbot.py


#Podešavanje stranice
st.set_page_config(
    page_title="AlexaGPT Chat",
    page_icon="🤖",
    layout="centered"
)


#Dark theme CSS
dark_css = """
<style>
    body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }

    [data-testid="stHeader"] {
        background-color: #0e1117 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        color: #e5e7eb !important;
    }

    /* Chat messages */
    .stChatMessage {
        color: #e5e7eb !important;
    }

    /* Chat input container */
    .stChatInputContainer {
        background-color: #111827 !important;
        padding: 12px;
        border-radius: 10px;
    }

    /* Chat input field */
    .stChatInputContainer textarea,
    .stChatInputContainer input {
        background-color: #1f2937 !important;
        color: #e5e7eb !important;
        border: 1px solid #374151 !important;
        border-radius: 6px !important;
    }
</style>
"""

st.markdown(dark_css, unsafe_allow_html=True)

# Session state za chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ćao, ja sam tvoj RAG chatbot. Postavi neko pitanje o Aleksi"}
    ]

st.title("🤖 AlexaGPT")
#st.caption("Dark mode • Streamlit UI • RAG backend")


#Prikaz istorije poruka
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


#Chat input
user_input = st.chat_input("Postavi pitanje...")

if user_input:
    # Dodaj korisničku poruku
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generiši odgovor
    with st.chat_message("assistant"):
        with st.spinner("Razmišljam..."):
            try:
                answer = rag(user_input)
            except Exception as e:
                answer = f"Došlo je do greške pri pozivu RAG-a: `{e}`"

            st.markdown(answer)

    # Sačuvaj odgovor
    st.session_state.messages.append({"role": "assistant", "content": answer})
