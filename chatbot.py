import os
import asyncio
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ──
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "").strip()

# Change 'us-east-1' to your actual Pinecone region if different
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
PORTFOLIO_MD = Path(__file__).parent / "portfolio.md"
INDEX_NAME = "portfolio-knowledge"

class HybridChatbot:
    def __init__(self):
        if not GOOGLE_API_KEY or not PINECONE_API_KEY:
            print("⚠️ API Keys missing! Chatbot will run in LLM-only mode.")
            self.rag_enabled = False
            return

        self.rag_enabled = True
        self.embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._history = {}

        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if INDEX_NAME not in existing_indexes:
                print(f"🚀 Creating Index in {PINECONE_REGION}...")
                self.pc.create_index(
                    name=INDEX_NAME,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
                )
                # Short wait to avoid timeout
                time.sleep(5)
            
            self.index = self.pc.Index(INDEX_NAME)
        except Exception as e:
            print(f"❌ Pinecone Init Error: {e}")
            self.rag_enabled = False

    def ingest_documents(self):
        if not self.rag_enabled or not PORTFOLIO_MD.exists(): return 0
        text = PORTFOLIO_MD.read_text(encoding="utf-8")
        chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80).split_text(text)
        
        vectors = []
        for i, chunk in enumerate(chunks):
            vectors.append({
                "id": f"c_{i}",
                "values": self.embeddings.embed_query(chunk),
                "metadata": {"text": chunk}
            })
        self.index.upsert(vectors=vectors)
        return len(chunks)

    def retrieve(self, query: str):
        if not self.rag_enabled: return []
        v = self.embeddings.embed_query(query)
        res = self.index.query(vector=v, top_k=3, include_metadata=True)
        return [m["metadata"]["text"] for m in res["matches"] if m["score"] > 0.35]

    async def respond(self, question: str, session_id: str = "default"):
        context = self.retrieve(question)
        context_text = "\n".join(context) if context else "No specific portfolio data found."
        
        prompt = f"Context about Bhavesh:\n{context_text}\n\nQuestion: {question}\nAnswer naturally:"
        resp = await asyncio.to_thread(self.llm.invoke, prompt)
        return {"answer": resp.content, "source": "hybrid" if context else "llm", "confidence": 0.8}

    def is_ready(self) -> bool:
        return self.rag_enabled
