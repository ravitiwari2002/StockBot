import streamlit as st
from logic import faq_questions

def apply_custom_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def display_chat_interface():
    st.markdown("## 💬 Stock Chatbot Assistant")

    user_input = st.text_area(
        "Send",
        height=150,
        placeholder="Ask me about stocks (e.g., 'What's the RSI of AAPL?')",
        label_visibility="collapsed",
        key="custom_text_input",
    )

    button_col = st.columns([4, 4, 2])[2]
    with button_col:
        submit_button = st.button("Send", use_container_width=True)

    return user_input, submit_button

def display_faqs():
    st.markdown("## ❓ FAQs")
    for question, answer in faq_questions:
        with st.expander(f"Q: {question}"):
            st.info(answer)
