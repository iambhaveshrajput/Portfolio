import os
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Key Loading ──
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "").strip()
INDEX_NAME = "portfolio-knowledge"
PORTFOLIO_MD = Path(__file__).parent / "portfolio.md"

class HybridChatbot:
    def __init__(self):
        if not GOOGLE_API_KEY or not PINECONE_API_KEY:
            raise OSError("API Keys missing in Render Environment settings.")

        self.embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.3)
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # ── Safe Index Creation ──
        try:
            if INDEX_NAME not in [idx.name for idx in self.pc.list_indexes()]:
                print(f"🚀 Creating index: {INDEX_NAME}")
                self.pc.create_index(
                    name=INDEX_NAME,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                while not self.pc.describe_index(INDEX_NAME).status['ready']:
                    time.sleep(2)
            self.index = self.pc.Index(INDEX_NAME)
        except Exception as e:
            print(f"⚠️ Index check/creation failed: {e}")

        # Auto-ingest if empty
        try:
            if self.index.describe_index_stats()['total_vector_count'] == 0:
                self.ingest_documents()
        except:
            pass

    def ingest_documents(self):
        if not PORTFOLIO_MD.exists(): return 0
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

    async def respond(self, question: str, session_id: str = "default"):
        v = self.embeddings.embed_query(question)
        res = self.index.query(vector=v, top_k=3, include_metadata=True)
        context = "\n".join([m["metadata"]["text"] for m in res["matches"] if m["score"] > 0.3])
        
        prompt = f"Context about Bhavesh:\n{context}\n\nQuestion: {question}"
        resp = await asyncio.to_thread(self.llm.invoke, [SystemMessage(content="You are a helpful portfolio assistant."), HumanMessage(content=prompt)])
        return {"answer": resp.content, "source": "hybrid", "confidence": 0.8}

    def is_ready(self):
        return True
