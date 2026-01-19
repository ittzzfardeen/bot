from dotenv import load_dotenv
import os
from src.helper import pdf_file_load , filter_to_minimal,text_split ,downloading_huggingface
from langchain_astradb import AstraDBVectorStore

load_dotenv()

groq_api_key=os.getenv("groq_api_key")
os.environ["groq_api_key"]="groq_api_key"


astra_db_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["astra_db_endpoint"]=astra_db_endpoint

ASTRA_DB_API_TOKEN=os.getenv("ASTRA_DB_API_TOKEN")
os.environ["ASTRA_DB_API_TOKEN"]=ASTRA_DB_API_TOKEN

minimal_data=pdf_file_load("/Users/fardeenkhan/coding/sec/data")
filter_data=filter_to_minimal(minimal_data)
text_chunks=text_split(filter_data)

embeddings=downloading_huggingface()


vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="my_docs_11",
  
    token=ASTRA_DB_API_TOKEN,
    api_endpoint=astra_db_endpoint
    
)
print("embedding strored........")