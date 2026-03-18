"""
RAG Engine - Core logic for Document Q&A using Endee Vector Database
"""

import os
import re
import uuid
from typing import List, Dict, Any

from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
from groq import Groq  # Free LLM API - can swap with any provider


# ── Configuration ──────────────────────────────────────────────────────────────
ENDEE_HOST     = os.getenv("ENDEE_HOST", "http://localhost:8080")
ENDEE_TOKEN    = os.getenv("ENDEE_AUTH_TOKEN", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")        # free at console.groq.com
INDEX_NAME     = "rag_documents"
EMBED_MODEL    = "all-MiniLM-L6-v2"                    # 384-dim, fast, free
EMBED_DIM      = 384
CHUNK_SIZE     = 500   # characters per chunk
CHUNK_OVERLAP  = 100
TOP_K          = 5     # chunks retrieved per query


# ── Helpers ────────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ── RAG Engine ─────────────────────────────────────────────────────────────────

class RAGEngine:
    def __init__(self):
        print("🔧  Loading sentence-transformer model …")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("🔗  Connecting to Endee …")
        self.client = Endee(ENDEE_TOKEN)
        self.client.set_base_url(f"{ENDEE_HOST}/api/v1")

        self._ensure_index()

        self.llm = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        if not self.llm:
            print("⚠️   GROQ_API_KEY not set – answers will show retrieved chunks only.")

    # ── Index management ───────────────────────────────────────────────────────

    def _ensure_index(self):
        """Create the Endee index if it doesn't already exist."""
        existing = [idx["name"] for idx in self.client.list_indexes()]
        if INDEX_NAME not in existing:
            print(f"📦  Creating Endee index '{INDEX_NAME}' …")
            self.client.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
        self.index = self.client.get_index(name=INDEX_NAME)
        print(f"✅  Connected to index '{INDEX_NAME}'")

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, source: str = "document") -> int:
        """Chunk text, embed, and upsert into Endee. Returns number of chunks stored."""
        chunks = chunk_text(text)
        vectors = self.embedder.encode(chunks, show_progress_bar=True).tolist()

        items = [
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "meta": {"text": chunk, "source": source, "chunk_index": i},
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self.index.upsert(items)
        print(f"✅  Stored {len(items)} chunks from '{source}'")
        return len(items)

    def ingest_file(self, filepath: str) -> int:
        """Read a .txt file and ingest it."""
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return self.ingest_text(text, source=os.path.basename(filepath))

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Embed query and fetch top-k matching chunks from Endee."""
        query_vec = self.embedder.encode([query])[0].tolist()
        results = self.index.query(vector=query_vec, top_k=top_k)
        return results

    # ── Generation ────────────────────────────────────────────────────────────

    def answer(self, question: str) -> Dict[str, Any]:
        """Full RAG pipeline: retrieve → build prompt → generate answer."""
        hits = self.retrieve(question)

        # Build context string from retrieved chunks
        context_parts = []
        sources = []
        for h in hits:
            meta = h.get("meta", {})
            context_parts.append(meta.get("text", ""))
            sources.append(meta.get("source", "unknown"))

        context = "\n\n---\n\n".join(context_parts)

        if self.llm:
            prompt = (
                "You are a helpful assistant. Use ONLY the context below to answer "
                "the question. If the answer is not in the context, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            chat = self.llm.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            answer_text = chat.choices[0].message.content.strip()
        else:
            # Fallback: just show the retrieved chunks
            answer_text = (
                "⚠️  No LLM configured. Here are the most relevant chunks:\n\n"
                + context
            )

        return {
            "question": question,
            "answer": answer_text,
            "sources": list(set(sources)),
            "num_chunks_retrieved": len(hits),
        }
