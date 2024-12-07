import streamlit as st
from config import MODEL_NAME  # ensure that this sets openai.api_key from .env
from ui import apply_custom_css, display_chat_interface, display_faqs
from logic import process_user_input

st.set_page_config(
    page_title="Stock Chatbot Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💬"
)

apply_custom_css()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

col1, col2 = st.columns(2, gap="medium")

with col1:
    user_input, submit_button = display_chat_interface()

    if submit_button and user_input:
        try:
            st.session_state['messages'] = process_user_input(user_input, st.session_state['messages'], MODEL_NAME)

            # Display conversation so far (optional)
            for msg in st.session_state['messages']:
                if msg['role'] == 'assistant':
                    st.write(msg['content'])

        except Exception as e:
            st.error("Oops! Something went wrong. Please try a different query or check your input.")
            st.error(f"An error occurred: {e}")

with col2:
    display_faqs()
