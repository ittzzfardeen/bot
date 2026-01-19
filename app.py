
from langchain_groq import ChatGroq


model = "llama-3.1-8b-instant"
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv("groq_api_key")
os.environ["groq_api_key"]="groq_api_key"

groq_model = ChatGroq(model=model, api_key=groq_api_key)
response = groq_model.invoke("What is the capital of India?").content 


print(response)