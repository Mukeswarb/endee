"""
Streamlit Web UI for RAG Document Q&A
Run: streamlit run app.py
"""

import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(
    page_title="RAG Doc Q&A — Powered by Endee",
    page_icon="🔍",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return RAGEngine()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG Document Q&A")
    st.markdown("**Powered by [Endee](https://endee.io) Vector Database**")
    st.divider()

    st.subheader("1️⃣  Ingest a Document")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file and st.button("📥 Ingest Document"):
        with st.spinner("Chunking & embedding …"):
            engine = get_engine()
            text = uploaded_file.read().decode("utf-8")
            n = engine.ingest_text(text, source=uploaded_file.name)
        st.success(f"✅ Stored {n} chunks from **{uploaded_file.name}**")

    st.divider()
    st.subheader("🧪 Quick Demo")
    if st.button("Load AI Knowledge Base"):
        demo_text = """
        Artificial Intelligence (AI) is the simulation of human intelligence in machines.
        Machine learning enables systems to learn from data without explicit programming.
        Deep learning uses neural networks to model complex patterns in data.
        Natural Language Processing (NLP) helps computers understand human language.
        Vector databases store high-dimensional vectors and enable similarity search.
        Endee is a high-performance vector database built for speed and scalability.
        Retrieval Augmented Generation (RAG) combines retrieval with language generation.
        """
        with st.spinner("Ingesting demo knowledge base …"):
            engine = get_engine()
            engine.ingest_text(demo_text, source="demo_ai_knowledge_base")
        st.success("✅ Demo knowledge base loaded!")

    st.divider()
    st.caption("🔗 [Endee GitHub](https://github.com/endee-io/endee)  |  [Docs](https://docs.endee.io)")


# ── Main Panel ─────────────────────────────────────────────────────────────────
st.title("🔍 Ask Questions About Your Documents")
st.markdown(
    "This app uses **Endee** as the vector database to store and retrieve document "
    "chunks, enabling accurate answers grounded in your own content."
)

question = st.text_input("💬 Enter your question:", placeholder="What is Retrieval Augmented Generation?")

if st.button("🔎 Get Answer", type="primary") and question:
    engine = get_engine()
    with st.spinner("Retrieving relevant chunks and generating answer …"):
        result = engine.answer(question)

    st.subheader("✅ Answer")
    st.write(result["answer"])

    with st.expander(f"📚 Retrieved {result['num_chunks_retrieved']} chunks from: {', '.join(result['sources'])}"):
        hits = engine.retrieve(question)
        for i, hit in enumerate(hits, 1):
            meta = hit.get("meta", {})
            score = hit.get("similarity", hit.get("score", "N/A"))
            st.markdown(f"**Chunk {i}** — Source: `{meta.get('source')}` | Similarity: `{score:.4f}`")
            st.info(meta.get("text", ""))

# ── Architecture Diagram ────────────────────────────────────────────────────────
with st.expander("🏗️ How It Works — System Architecture"):
    st.markdown("""
    ```
    ┌──────────────┐     chunk + embed      ┌─────────────────────┐
    │  Your .txt   │ ──────────────────────▶ │   Endee Vector DB   │
    │  Document    │                         │  (HNSW Index, cosine│
    └──────────────┘                         │   similarity, INT8) │
                                             └──────────┬──────────┘
                                                        │
    ┌──────────────┐     embed query                    │ top-k chunks
    │  User Query  │ ──────────────────────────────────▶│
    └──────────────┘                                    ▼
                                             ┌─────────────────────┐
                                             │   LLM (Groq/llama3) │
                                             │  Answer grounded in │
                                             │  retrieved context  │
                                             └─────────────────────┘
    ```
    **Components:**
    - **Endee** — Vector store for document embeddings
    - **sentence-transformers** — `all-MiniLM-L6-v2` for creating 384-dim embeddings
    - **Groq (llama3-8b)** — Free, fast LLM for answer generation
    - **Streamlit** — Web interface
    """)
