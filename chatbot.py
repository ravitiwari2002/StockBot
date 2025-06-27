import json
import openai
import streamlit as st
from functions import *

openai.api_key = st.secrets["OPENAI_API_KEY"]

MODEL_NAME = 'gpt-3.5-turbo-0125'
functions_relative_path = 'function_config.json'

with open(functions_relative_path, 'r') as file:
    config_data = json.load(file)

def run_chatbot():
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
        (
        "What type of financial information does the chatbot provide?", "The chatbot provides a comprehensive overview "
                                                                        "of stock market data. It delivers real-time and "
                                                                        "historical prices, daily returns, "
                                                                        "technical indicators, comparative analysis, "
                                                                        "trading volume, dividend details, "
                                                                        "and the latest news."),
        ("Is the chatbot limited to a specific set of stocks?",
         "No, the chatbot can provide information on a wide range "
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
                # Split user input into separate questions
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

                    if isinstance(response_message, dict):
                        if response_message.get('function_call'):
                            function_name = response_message['function_call']['name']
                            function_args = json.loads(response_message['function_call']['arguments'])
                            # Check if the function requires a 'ticker' argument
                            if function_name in ['get_stock_price', 'calculate_RSI', 'calculate_MACD',
                                                 'plot_stock_price',
                                                 'get_stock_news', 'calculate_daily_returns']:
                                args_dict = {'ticker': function_args.get('ticker')}
                            elif function_name in ['calculate_SMA', 'calculate_EMA']:
                                args_dict = {'ticker': function_args.get('ticker'),
                                             'window': function_args.get('window')}
                            elif function_name == 'compare_stock_prices':
                                args_dict = {'ticker1': function_args.get('ticker1'),
                                             'ticker2': function_args.get('ticker2'),
                                             'period': function_args.get('period')}
                            elif function_name == 'average_volume':
                                args_dict = {'ticker': function_args.get('ticker'),
                                             'period': function_args.get('period')}
                            elif function_name == 'get_dividend_info':
                                args_dict = {'ticker': function_args.get('ticker')}

                            function_to_call = available_functions[function_name]
                            function_response = function_to_call(**args_dict)

                            if function_name == 'plot_stock_price':
                                st.image('stock.png')

                            elif function_name == 'get_stock_news':
                                st.text("Recent News Headlines:")
                                for article in function_response:
                                    st.write(f"Title: {article['title']}")
                                    st.write(f"Publisher: {article['publisher']}")
                                    st.write(f"Link: {article['link']}")
                                    publish_time = datetime.utcfromtimestamp(article['providerPublishTime']).strftime(
                                        '%Y-%m-%d %H:%M:%S')
                                    st.write(f"Provider Publish Time: {publish_time}")
                                    st.write(f"Type: {article['type']}")
                                    if 'thumbnail' in article and 'resolutions' in article['thumbnail'] and \
                                            article['thumbnail']['resolutions']:
                                        st.image(article['thumbnail']['resolutions'][0]['url'], width=200)
                                    individual_tickers = [ticker for ticker in article['relatedTickers'] if
                                                          not (ticker.startswith('^') or ticker.endswith('=F'))]
                                    st.write("Related Tickers: ", ", ".join(individual_tickers))
                                    st.write("\n---\n")

                            elif function_name == 'calculate_daily_returns':
                                daily_returns = pd.Series(
                                    {k: (float(v) if v != "NaN" else np.nan) for k, v in function_response.items()})
                                st.write(daily_returns)

                            else:
                                st.session_state['messages'].append(response_message)
                                st.session_state['messages'].append(
                                    {
                                        'role': 'function',
                                        'name': function_name,
                                        'content': function_response
                                    }
                                )
                                second_response = openai.ChatCompletion.create(
                                    model=MODEL_NAME,
                                    messages=st.session_state['messages']
                                )
                                st.write(second_response['choices'][0]['message']['content'])
                                st.session_state['messages'].append(
                                    {'role': 'assistant',
                                     'content': second_response['choices'][0]['message']['content']})

                        else:
                            st.write(response_message['content'])
                            st.session_state['messages'].append(
                                {'role': 'assistant', 'content': response_message['content']})
                    else:
                        st.write(response_message)
                        st.session_state['messages'].append({'role': 'assistant', 'content': response_message})
            except Exception as e:
                st.error("Oops! Something went wrong. Please try a different query or check your input.")
                st.error(f"An error occurred: {e}")

    # FAQ Section
    with col2:
        st.markdown("## ❓ FAQs")
        for question, answer in faq_questions:
            with st.expander(f"Q: {question}"):
                st.info(answer)

    # Adjust layout colors
    st.markdown(
        """
        <style>
            .st-df div { color: #333; }
            .st-df td, .st-df th { border: 1px solid #ddd !important; }
        </style>
        """,
        unsafe_allow_html=True
    )
