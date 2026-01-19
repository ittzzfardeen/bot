from flask import Flask,render_template,jsonify,request
from src.helper import downloading_huggingface
from langchain_astradb import AstraDBVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from src.prompt import *
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

app=Flask(__name__)

load_dotenv()



astra_db_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["astra_db_endpoint"]=astra_db_endpoint

ASTRA_DB_API_TOKEN=os.getenv("ASTRA_DB_API_TOKEN")
os.environ["ASTRA_DB_API_TOKEN"]=ASTRA_DB_API_TOKEN

groq_api_key=os.getenv("groq_api_key")
os.environ["groq_api_key"]=groq_api_key

embeddings=downloading_huggingface()

vectorstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="my_docs_11",
  
    token=ASTRA_DB_API_TOKEN,
    api_endpoint=astra_db_endpoint
    
)
retriever = vectorstore.as_retriever()
model="llama-3.1-8b-instant"
groq_model=ChatGroq(model=model,api_key=groq_api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(
            d.page_content for d in docs)
rag_chain = (
    {
        "context": retriever|format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | groq_model
)

@app.route("/")
def index():
    return render_template('chat.html')
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)