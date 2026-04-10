import os
import asyncio
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone, ServerlessSpec
# Updated imports for LangChain 0.3+
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Key Loading ──
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "").strip()

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise OSError("API Keys missing in Render settings. Check the 'Environment' tab.")

PORTFOLIO_MD = Path(__file__).parent / "portfolio.md"
# Using 2026 state-of-the-art embedding model
EMBEDDING_MODEL = "text-embedding-004"
INDEX_NAME = "portfolio-knowledge"

class HybridChatbot:
    def __init__(self):
        # Initializing models
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, 
            google_api_key=GOOGLE_API_KEY
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=GOOGLE_API_KEY, 
            temperature=0.3
        )

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Automatic index management
        if INDEX_NAME not in [idx.name for idx in self.pc.list_indexes()]:
            print(f"🚀 Creating Pinecone Index: {INDEX_NAME}...")
            self.pc.create_index(
                name=INDEX_NAME, 
                dimension=768, # text-embedding-004 uses 768 dims
                metric="cosine", 
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not self.pc.describe_index(INDEX_NAME).status['ready']:
                time.sleep(2)

        self.index = self.pc.Index(INDEX_NAME)
        self._history = {}

        # Initial data check
        stats = self.index.describe_index_stats()
        if stats['total_vector_count'] == 0:
            print("📚 First-time deploy: Ingesting portfolio data...")
            self.ingest_documents()

    def ingest_documents(self):
        if not PORTFOLIO_MD.exists(): 
            print("⚠️ Warning: portfolio.md not found.")
            return 0
        
        text = PORTFOLIO_MD.read_text(encoding="utf-8")
        # Updated text splitter for LangChain 0.3
        char_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        chunks = char_splitter.split_text(text)
        
        vectors = []
        for i, chunk in enumerate(chunks):
            vectors.append({
                "id": f"chunk_{i}",
                "values": self.embeddings.embed_query(chunk),
                "metadata": {"text": chunk}
            })
        
        self.index.upsert(vectors=vectors)
        print(f"✅ Ingested {len(chunks)} chunks to Pinecone.")
        return len(chunks)

    def retrieve(self, query: str):
        v = self.embeddings.embed_query(query)
        res = self.index.query(vector=v, top_k=3, include_metadata=True)
        return [m["metadata"]["text"] for m in res["matches"] if m["score"] > 0.3]

    async def respond(self, question: str, session_id: str = "default"):
        context_chunks = self.retrieve(question)
        context_text = "\n".join(context_chunks)
        
        sys_msg = SystemMessage(content=(
            f"You are a helpful AI assistant for Bhavesh Rajput. Answer using this context:\n"
            f"{context_text}"
        ))
        user_msg = HumanMessage(content=question)
        
        resp = await asyncio.to_thread(self.llm.invoke, [sys_msg, user_msg])
        return {"answer": resp.content, "source": "hybrid", "confidence": 0.8}

    def is_ready(self):
        return True
