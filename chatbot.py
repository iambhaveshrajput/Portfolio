import os
import asyncio
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ── Key Validation (Tells you exactly which one is missing) ──────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not GOOGLE_API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY is missing from environment variables!")
if not PINECONE_API_KEY:
    print("❌ ERROR: PINECONE_API_KEY is missing from environment variables!")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise OSError("Missing API Keys. Please check your Render Environment settings.")

PORTFOLIO_MD = Path(__file__).parent / "portfolio.md"
INDEX_NAME   = "portfolio-knowledge"
CHUNK_SIZE   = 600
CHUNK_OVERLAP= 80
TOP_K        = 4
MIN_SCORE    = 0.35 

class HybridChatbot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )
        self.classifier_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # ── Robust Index Creation ──
        if INDEX_NAME not in self.pc.list_indexes().names():
            print(f"🚀 Creating new Pinecone index: {INDEX_NAME}...")
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=768, 
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            # Wait for index to be ready (Essential for first-time deploy)
            while not self.pc.describe_index(INDEX_NAME).status['ready']:
                print("⏳ Waiting for Pinecone index to initialize...")
                time.sleep(5)

        self.index = self.pc.Index(INDEX_NAME)

        # Only ingest if index is empty
        stats = self.index.describe_index_stats()
        if stats['total_vector_count'] == 0 and PORTFOLIO_MD.exists():
            print("📚 Index is empty. Auto-ingesting portfolio.md …")
            self.ingest_documents()

        self._history = {}

    def ingest_documents(self) -> int:
        if not PORTFOLIO_MD.exists():
            return 0
        text = PORTFOLIO_MD.read_text(encoding="utf-8")
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
        char_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = char_splitter.split_documents(header_splitter.split_text(text))
        
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.embeddings.embed_query(chunk.page_content)
            vectors.append({
                "id": f"chunk_{i}",
                "values": embedding,
                "metadata": {"text": chunk.page_content, "source": "portfolio.md", **chunk.metadata}
            })

        self.index.upsert(vectors=vectors)
        return len(chunks)

    def retrieve(self, query: str) -> list[str]:
        query_vector = self.embeddings.embed_query(query)
        results = self.index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)
        return [match["metadata"]["text"] for match in results["matches"] if match["score"] > MIN_SCORE]

    async def classify(self, question: str) -> dict:
        msgs = [SystemMessage(content=CLASSIFIER_SYSTEM), HumanMessage(content=f"Question: {question}")]
        resp = await asyncio.to_thread(self.llm.invoke, msgs)
        raw  = re.sub(r"```json|```","", resp.content.strip()).strip()
        try:
            return json.loads(raw)
        except:
            return {"mode":"hybrid"}

    async def respond(self, question: str, session_id: str = "default") -> dict:
        classification = await self.classify(question)
        mode = classification.get("mode", "hybrid")
        context_chunks = self.retrieve(question) if mode in ("rag", "hybrid") else []
        context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
        
        sys_msg = RAG_SYSTEM.format(context=context_text) if mode == "rag" else HYBRID_SYSTEM.format(context=context_text) if mode == "hybrid" else LLM_SYSTEM
        
        resp = await asyncio.to_thread(self.llm.invoke, [SystemMessage(content=sys_msg), HumanMessage(content=question)])
        return {"answer": resp.content.strip(), "source": mode, "confidence": 0.8}

    def is_ready(self) -> bool:
        return self.index.describe_index_stats()['total_vector_count'] > 0
