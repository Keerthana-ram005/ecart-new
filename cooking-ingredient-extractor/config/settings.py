import os
from dotenv import load_dotenv

load_dotenv()

# Configuration settings
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "outputs/models")
LOG_PATH = os.getenv("LOG_PATH", "outputs/logs")
