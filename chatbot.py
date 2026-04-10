"""
HybridChatbot — RAG + LLM with Gemini 1.5 Flash classifier
API key is read ONLY from the GOOGLE_API_KEY environment variable.
Never hardcode it — set it in .env or your deployment environment.
"""

import os
import asyncio
import json
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ── Config ──────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY environment variable is not set.\n"
        "Create a .env file with:  GOOGLE_API_KEY=your_key_here\n"
        "Or export it:             export GOOGLE_API_KEY=your_key_here"
    )

PORTFOLIO_MD = Path(__file__).parent / "portfolio.md"
CHROMA_PATH  = Path(__file__).parent / "chroma_db"
COLLECTION   = "portfolio_knowledge"
CHUNK_SIZE   = 600
CHUNK_OVERLAP= 80
TOP_K        = 4
MIN_SCORE    = 0.35

# ── Prompts ──────────────────────────────────────────────────────────────────
CLASSIFIER_SYSTEM = """You are a query classifier for a portfolio chatbot.
Given a user question, decide how to answer it:
- "rag"    → specifically about THIS person's portfolio (projects, skills, experience, bio, contact, education)
- "llm"    → generic / technical / world knowledge
- "hybrid" → needs BOTH portfolio context AND general knowledge

Reply ONLY with valid JSON: {"mode": "rag", "reasoning": "..."}
No other text."""

RAG_SYSTEM = """You are a friendly assistant for a personal portfolio website.
Answer using ONLY the portfolio context below. If context is insufficient, say so.
Keep answers concise and professional. Use markdown when helpful.

PORTFOLIO CONTEXT:
{context}"""

LLM_SYSTEM = """You are a helpful assistant embedded in a developer's portfolio website.
Answer using your general knowledge. Keep answers concise. Use markdown when helpful."""

HYBRID_SYSTEM = """You are a knowledgeable assistant for a personal portfolio website.
Use BOTH the portfolio context AND your general knowledge for a rich answer.

PORTFOLIO CONTEXT:
{context}"""


class HybridChatbot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True,
        )
        self.classifier_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
            convert_system_message_to_human=True,
        )
        self.ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GOOGLE_API_KEY,
            model_name="models/embedding-001",
        )
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        if self.collection.count() == 0 and PORTFOLIO_MD.exists():
            print("📚 Auto-ingesting portfolio.md …")
            self.ingest_documents()

        self._history: dict[str, list[dict]] = {}

    def ingest_documents(self) -> int:
        if not PORTFOLIO_MD.exists():
            raise FileNotFoundError(f"portfolio.md not found at {PORTFOLIO_MD}")
        text = PORTFOLIO_MD.read_text(encoding="utf-8")
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")],
            strip_headers=False,
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n","\n",". "," "],
        )
        chunks = char_splitter.split_documents(header_splitter.split_text(text))
        if not chunks:
            raise ValueError("No chunks produced from portfolio.md")
        self.collection.delete(where={"source": "portfolio.md"})
        self.collection.upsert(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=[c.page_content for c in chunks],
            metadatas=[{"source":"portfolio.md","index":i,**c.metadata} for i,c in enumerate(chunks)],
        )
        print(f"✅ Ingested {len(chunks)} chunks")
        return len(chunks)

    def retrieve(self, query: str) -> list[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=min(TOP_K, self.collection.count() or 1),
            include=["documents","distances"],
        )
        return [
            doc for doc, dist in zip(results["documents"][0], results["distances"][0])
            if dist < (1 - MIN_SCORE)
        ]

    async def classify(self, question: str) -> dict:
        msgs = [SystemMessage(content=CLASSIFIER_SYSTEM), HumanMessage(content=f"Question: {question}")]
        resp = await asyncio.to_thread(self.classifier_llm.invoke, msgs)
        raw  = re.sub(r"```json|```","", resp.content.strip()).strip()
        try:
            result = json.loads(raw)
            if result.get("mode") not in ("rag","llm","hybrid"):
                result["mode"] = "hybrid"
        except json.JSONDecodeError:
            result = {"mode":"hybrid","reasoning":"parse error"}
        return result

    async def respond(self, question: str, session_id: str = "default") -> dict:
        classification = await self.classify(question)
        mode = classification["mode"]

        context_chunks = []
        if mode in ("rag","hybrid"):
            context_chunks = self.retrieve(question)
            if not context_chunks and mode == "rag":
                mode = "llm"

        context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
        if mode == "rag":
            sys = RAG_SYSTEM.format(context=context_text)
        elif mode == "hybrid":
            sys = HYBRID_SYSTEM.format(context=context_text)
        else:
            sys = LLM_SYSTEM

        history = self._history.get(session_id, [])
        msgs = [SystemMessage(content=sys)]
        for h in history[-6:]:
            msgs.append(HumanMessage(content=h["content"]) if h["role"]=="user"
                        else SystemMessage(content=f"[Previous answer]: {h['content']}"))
        msgs.append(HumanMessage(content=question))

        resp   = await asyncio.to_thread(self.llm.invoke, msgs)
        answer = resp.content.strip()

        self._history.setdefault(session_id, [])
        self._history[session_id] += [
            {"role":"user","content":question},
            {"role":"assistant","content":answer},
        ]
        return {"answer": answer, "source": mode, "confidence": 0.9 if context_chunks else 0.7}

    def is_ready(self) -> bool:
        return self.collection.count() > 0
