import openai
import pandas as pd
import numpy as np
from datetime import datetime

faq_questions = [
    ("What can I ask the chatbot about?",
     "Feel free to inquire about a wide range of stock-related information, including current stock prices, daily returns, technical indicators, comparisons of stock prices, average trading volume, dividend details, and the latest news."),
    ("How do I interact with the chatbot?",
     "Type your stock-related questions in the input box and press Enter. The chatbot will respond with relevant information."),
    ("Can I get historical stock data using this chatbot?",
     "Yes, you can ask for historical stock prices, moving averages, and other technical indicators over specific periods."),
    ("What type of financial information does the chatbot provide?",
     "The chatbot provides a comprehensive overview of stock market data: real-time and historical prices, daily returns, technical indicators, comparative analysis, trading volume, dividend details, and the latest news."),
    ("Is the chatbot limited to a specific set of stocks?",
     "No, the chatbot can provide information on a wide range of stocks. Just specify the stock ticker symbol."),
    ("What are the limitations of the chatbot?",
     "The chatbot might not have data for all stocks, and the info provided may not be real-time. Also, it may not cover every financial metric. One query at a time is recommended for best results.")
]

def process_user_input(user_input, session_messages, model_name):
    """
    Process the user input by sending it to OpenAI and updating the session messages.
    """
    user_questions = user_input.split('\n')
    for user_question in user_questions:
        session_messages.append({'role': 'user', 'content': user_question})
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=session_messages
        )

        response_message = response['choices'][0]['message']

        if isinstance(response_message, dict):
            session_messages.append({'role': 'assistant', 'content': response_message.get('content', '')})
        else:
            session_messages.append({'role': 'assistant', 'content': str(response_message)})

    return session_messages
