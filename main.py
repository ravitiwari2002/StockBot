import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import both features
from chatbot import run_chatbot
from forecast import run_forecast

st.set_page_config(page_title="Stock Assistant", layout="wide")

st.sidebar.title("📊 Stock Assistant")
option = st.sidebar.radio("Select a feature", ["💬 Chatbot", "📈 Forecasting"])

if option == "💬 Chatbot":
    run_chatbot()
else:
    run_forecast()
