import os  # Added missing import
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import HybridChatbot
import uvicorn

app = FastAPI(title="Portfolio Hybrid Chatbot API")

# Middleware configuration
# Updated origins list to include both of your Netlify domains
origins = [
    "https://bhaveshrajputportfolio.netlify.app", # The link you confirmed
    "https://outportfolio.netlify.app",            
    "http://localhost:3000",                       # For local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Required for secure communication with specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot instance
# Note: The new chatbot.py handles Pinecone connection in __init__
chatbot = HybridChatbot()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    source: str
    confidence: float

@app.get("/")
def root():
    return {"status": "Portfolio Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # chatbot.respond is an async function in your latest chatbot.py
        result = await chatbot.respond(request.message, request.session_id)
        return result
    except Exception as e:
        # Standard error logging for your Render logs
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents():
    """Manually trigger data ingestion from portfolio.md to Pinecone Cloud."""
    try:
        count = chatbot.ingest_documents()
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    # Checks if the Pinecone index is ready and has data
    return {"status": "healthy", "rag_ready": chatbot.is_ready()}

if __name__ == "__main__":
    # Render assigns a dynamic port; this logic ensures your app binds to it correctly
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
