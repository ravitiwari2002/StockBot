import json
import openai
import streamlit as st
from functions import *
import os

current_directory = os.getcwd()
api_relative_path = 'API_KEY'
functions_relative_path = 'function_config.json'

api_key_path = os.path.join(current_directory, api_relative_path)
MODEL_NAME = 'gpt-3.5-turbo-0125'

with open(api_key_path, 'r') as f:
    openai.api_key = f.read()

config_file_path = os.path.join(current_directory, functions_relative_path)
with open(config_file_path, 'r') as file:
    config_data = json.load(file)

available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price,
    'compare_stock_prices': compare_stock_prices,
    'average_volume': average_volume,
    'get_dividend_info': get_dividend_info,
    'get_stock_news': get_stock_news,
    'calculate_daily_returns': calculate_daily_returns
}

faq_questions = [
    ("What can I ask the chatbot about?", "Feel free to inquire about a wide range of stock-related information, "
                                          "including current stock prices, daily returns, technical indicators, "
                                          "comparisons of stock prices, average trading volume, dividend details, "
                                          "and the latest news."),
    ("How do I interact with the chatbot?", "Type your stock-related questions in the input box provided and press "
                                            "Enter. The chatbot will respond with relevant information."),
    ("Can I get historical stock data using this chatbot?", "Yes, you can ask for historical stock prices, moving "
                                                            "averages, and other technical indicators over specific "
                                                            "periods."),
    ("What type of financial information does the chatbot provide?", "The chatbot provides a comprehensive overview "
                                                                     "of stock market data. It delivers real-time and "
                                                                     "historical prices, daily returns, "
                                                                     "technical indicators, comparative analysis, "
                                                                     "trading volume, dividend details, "
                                                                     "and the latest news."),
    ("Is the chatbot limited to a specific set of stocks?", "No, the chatbot can provide information on a wide range "
                                                            "of stocks. You can inquire about any publicly traded "
                                                            "company by specifying its stock ticker symbol."),
    ("What are the limitations of the chatbot?",
     "The chatbot is designed to offer information on publicly traded stocks. However, please note that it might not "
     "have data for all stocks, and the information provided may not be real-time. Additionally, it may not cover "
     "every financial metric or event related to stocks. We recommend entering one query at a time for the best "
     "results.")
]


# Set Streamlit layout configuration
st.set_page_config(
    page_title="Stock Chatbot Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💬"
)

# Custom CSS for page layout
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to bottom, #1e3c72, #2a5298);
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            font-size: 16px;
            padding: 8px 20px;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #2575fc, #6a11cb);
            transform: scale(1.05);
        }
        .stTextInput>div>div>textarea {
            background-color: #f7f9fc;
            border: 1px solid #ccc;
            border-radius: 8px;
            color: #333;
            padding: 10px;
        }
        .st-df div { color: #333; }
        .st-df td, .st-df th {
            border: 1px solid #ddd !important;
        }
        .st-expander {
            background-color: #f5f5f5;
            border-radius: 8px;
            padding: 10px;
        }
        .st-expander-header {
            font-weight: bold;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Main layout
col1, col2 = st.columns(2, gap="medium")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Chatbot Section
with col1:
    st.markdown("## 💬 Stock Chatbot Assistant")

    # Custom CSS to style the text area and button
    st.markdown(
        """
        <style>
        .custom-text-area {
            width: 100%;
            padding: 10px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background-color: #f7f9fc;
            color: #333;
            resize: none;
            outline: none; /* Removes focus outline */
        }
        .custom-button {
            margin-top: 10px;
            padding: 10px 20px; /* Adjust padding for comfortable size */
            font-size: 16px;
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .custom-button:hover {
            background: linear-gradient(to right, #2575fc, #6a11cb);
            transform: scale(1.05);
        }
        .stTextInput > div {
            display: none !important; /* Hide the default outer container */
        }
        .no-ctrl-enter-hint textarea {
            caret-color: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Input layout
    user_input = st.text_area(
        "Send",
        height=150,
        placeholder="Ask me about stocks (e.g., 'What's the RSI of AAPL?')",
        label_visibility="collapsed",
        key="custom_text_input",
    )

    # Position the button below the text area
    button_col = st.columns([4, 4, 2])[2]  # Adjust columns for button alignment
    with button_col:
        submit_button = st.button("Send", use_container_width=True)

    # Process input when the button is clicked
    if submit_button and user_input:
        try:
            user_questions = user_input.split('\n')
            for user_question in user_questions:
                st.session_state['messages'].append({'role': 'user', 'content': user_question})

                response = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    messages=st.session_state['messages'],
                    functions=config_data,
                    function_call='auto'
                )

                response_message = response['choices'][0]['message']

                if isinstance(response_message, dict) and response_message.get('function_call'):
                    function_name = response_message['function_call']['name']
                    function_args = json.loads(response_message['function_call']['arguments'])
                    args_dict = {k: function_args.get(k) for k in function_args}

                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**args_dict)

                    if function_name == 'plot_stock_price':
                        st.image('stock.png')
                    elif function_name == 'get_stock_news':
                        st.markdown("### 📰 Recent News Headlines")
                        for article in function_response:
                            st.markdown(f"**Title**: {article['title']}")
                            st.markdown(f"**Publisher**: {article['publisher']}")
                            st.markdown(f"[Read More]({article['link']})")
                            st.write("---")
                    else:
                        st.write(function_response)
                else:
                    st.write(response_message.get('content', 'No content returned.'))

        except Exception as e:
            st.error("Oops! Something went wrong. Please try a different query or check your input.")
            st.error(f"Error: {e}")

# FAQ Section
with col2:
    st.markdown("## ❓ FAQs")
    for question, answer in faq_questions:
        with st.expander(f"Q: {question}"):
            st.info(answer)
