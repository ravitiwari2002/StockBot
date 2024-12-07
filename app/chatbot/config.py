import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = 'gpt-3.5-turbo-0125'

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
