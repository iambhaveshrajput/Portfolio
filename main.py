__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# 1. Grab the API key from the cloud hosting environment
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# 2. On-Startup: Auto-Ingest the .md file and create the Vector Store
def setup_retriever():
    print("Initializing Database...")
    with open("my_info.md", "r", encoding="utf-8") as f:
        content = f.read()
    
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = splitter.split_text(content)
    
    # Creates an in-memory Chroma database
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = setup_retriever()

# 3. Shared Memory for the last 5 interactions
memory = ConversationBufferWindowMemory(k=5, return_messages=True, memory_key="chat_history")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.message
    history = memory.load_memory_variables({})["chat_history"]

    # Step A: Intent Classifier (The Router)
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "Output 'RAG' if the question is about Bhavesh Rajput, his projects, skills, education, or background. Otherwise output 'GENERAL'."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    route_chain = router_prompt | llm | StrOutputParser()
    route = route_chain.invoke({"input": user_input, "chat_history": history})

    # Step B: Route Execution
    if "RAG" in route:
        docs = retriever.invoke(user_input)
        context = "\n".join([d.page_content for d in docs])
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are Bhavesh's professional portfolio assistant. Answer questions using only this context: {context}. Be concise and polite."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful technical AI assistant on a software engineer's portfolio. Answer concisely."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

    chain = prompt | llm
    response = chain.invoke({"input": user_input, "chat_history": history})
    
    # Save the new interaction to memory
    memory.save_context({"input": user_input}, {"output": response.content})
    
    return {"response": response.content}
