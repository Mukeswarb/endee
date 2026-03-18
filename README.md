# 📄 RAG Document Q&A — Powered by Endee Vector Database

A beginner-friendly **Retrieval Augmented Generation (RAG)** application that lets you upload any text document and ask natural language questions about it. Built with **[Endee](https://github.com/endee-io/endee)** as the vector database.

---

## 🎯 Project Overview

This project demonstrates a practical RAG pipeline where:

1. You upload a `.txt` document
2. The text is split into chunks and embedded using `sentence-transformers`
3. Embeddings are stored in **Endee**, a high-performance vector database
4. When you ask a question, it is embedded and used to retrieve the most relevant chunks from Endee
5. The retrieved chunks are passed to a free LLM (Groq/llama3) to generate a grounded answer

**Use Cases:**
- Q&A over private documents (notes, manuals, reports)
- Knowledge base chatbot
- Study assistant

---

## 🏗️ System Design

```
┌──────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                   │
│                                                          │
│  .txt File  ──▶  Chunker  ──▶  Embedder  ──▶  Endee DB  │
│             (500 char,       (MiniLM-L6,    (HNSW index, │
│              100 overlap)     384-dim)       cosine sim) │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                      │
│                                                          │
│  User Query ──▶  Embedder ──▶  Endee Search (top-k=5)   │
│                                        │                 │
│                                        ▼                 │
│                              Retrieved Chunks            │
│                                        │                 │
│                                        ▼                 │
│                           LLM (llama3-8b via Groq)       │
│                                        │                 │
│                                        ▼                 │
│                              Grounded Answer             │
└──────────────────────────────────────────────────────────┘
```

### Components

| Component | Role |
|---|---|
| **Endee** | Vector database — stores and retrieves document embeddings |
| **sentence-transformers** | `all-MiniLM-L6-v2` model for 384-dim text embeddings |
| **Groq (llama3-8b-8192)** | Free, fast LLM for answer generation |
| **Streamlit** | Web interface |
| **Python CLI** | Lightweight command-line interface |

---

## 🗂️ Project Structure

```
rag-docqa-endee/
├── rag_engine.py        # Core RAG logic: chunking, embedding, Endee integration, LLM
├── main.py              # CLI interface (ingest / ask / demo)
├── app.py               # Streamlit web UI
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── sample_docs/
│   └── ai_knowledge_base.txt   # Sample document for testing
└── README.md
```

---

## 🔑 How Endee Is Used

Endee serves as the **core vector store** of this project. Here is exactly how it is integrated:

### 1. Connecting to Endee
```python
from endee import Endee, Precision

client = Endee(auth_token)          # connect to local Endee server
client.set_base_url("http://localhost:8080/api/v1")
```

### 2. Creating an Index
```python
client.create_index(
    name="rag_documents",
    dimension=384,           # matches MiniLM-L6-v2 output size
    space_type="cosine",     # cosine similarity for semantic search
    precision=Precision.INT8 # quantized for efficiency
)
```

### 3. Storing Embeddings (Upsert)
```python
index = client.get_index(name="rag_documents")
index.upsert([
    {
        "id": "chunk-uuid",
        "vector": [0.12, 0.45, ...],   # 384-dim embedding
        "meta": {"text": "chunk text", "source": "file.txt"}
    }
])
```

### 4. Semantic Search (Query)
```python
results = index.query(vector=query_embedding, top_k=5)
# returns top-5 most similar chunks with similarity scores
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (to run Endee)
- Free Groq API key from [console.groq.com](https://console.groq.com) *(optional but recommended)*

---

### Step 1 — Star & Fork Endee ⭐

> **Required by the assignment.**

1. Go to [https://github.com/endee-io/endee](https://github.com/endee-io/endee)
2. Click **⭐ Star**
3. Click **Fork** → your GitHub account

---

### Step 2 — Clone This Repository

```bash
git clone https://github.com/<your-username>/rag-docqa-endee.git
cd rag-docqa-endee
```

---

### Step 3 — Start Endee (Docker)

```bash
mkdir endee && cd endee

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
services:
  endee:
    image: endeeio/endee-server:latest
    container_name: endee-server
    ports:
      - "8080:8080"
    environment:
      NDD_NUM_THREADS: 0
      NDD_AUTH_TOKEN: ""
    volumes:
      - endee-data:/data
    restart: unless-stopped
volumes:
  endee-data:
EOF

docker compose up -d
cd ..
```

Verify it's running: open [http://localhost:8080](http://localhost:8080)

---

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 5 — Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

---

### Step 6 — Run the App

**Option A — Web UI (Streamlit)**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**Option B — Command Line**
```bash
# Run a quick demo (no file needed)
python main.py demo

# Ingest your own document
python main.py ingest sample_docs/ai_knowledge_base.txt

# Ask a question
python main.py ask "What is Retrieval Augmented Generation?"
```

---

## 📸 Demo

```
❓ Question : What is Retrieval Augmented Generation?
📚 Sources  : ai_knowledge_base.txt
🔢 Chunks   : 5 retrieved

💬 Answer:
Retrieval Augmented Generation (RAG) is an AI framework that improves LLM responses
by grounding them in external knowledge. It retrieves relevant document chunks from a
vector database (like Endee) and provides them as context to the LLM, reducing
hallucinations and enabling answers about private or recent data.
```

---

## 🧠 Key Design Decisions

- **Endee over other vector DBs** — Fast HNSW indexing, simple Python SDK, Docker-first, open source
- **INT8 precision** — Reduces memory footprint with minimal accuracy loss
- **Overlapping chunks** — 100-char overlap preserves context across chunk boundaries
- **Cosine similarity** — Better than dot product for normalized text embeddings
- **Groq (free tier)** — Fast LLM inference, no credit card required

---

## 📝 License

MIT License — free to use and modify.
