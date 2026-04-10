from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import HybridChatbot
import uvicorn

app = FastAPI(title="Portfolio Hybrid Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your portfolio domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = HybridChatbot()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    source: str  # "rag" | "llm" | "hybrid"
    confidence: float

@app.get("/")
def root():
    return {"status": "Portfolio Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = await chatbot.respond(request.message, request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_documents():
    """Re-ingest the portfolio.md file into ChromaDB."""
    try:
        count = chatbot.ingest_documents()
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "rag_ready": chatbot.is_ready()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
