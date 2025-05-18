from dotenv import load_dotenv
import os
import json
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
key= os.getenv("key")
secret= os.getenv("secret")
google_api= os.getenv("google_api_ai")