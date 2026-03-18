"""
CLI — Interact with the RAG Document Q&A system
Usage:
    python main.py ingest <file.txt>
    python main.py ask "Your question here"
    python main.py demo
"""

import sys
from rag_engine import RAGEngine


DEMO_TEXT = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines
that are programmed to think and learn. Machine learning is a subset of AI that
enables systems to learn from data without being explicitly programmed.

Deep learning uses neural networks with many layers to model complex patterns in data.
It has revolutionized image recognition, natural language processing, and speech recognition.

Natural Language Processing (NLP) is a branch of AI focused on enabling computers to
understand, interpret, and generate human language. Applications include chatbots,
translation, and sentiment analysis.

Vector databases store high-dimensional vectors and enable similarity search.
They are foundational to modern AI applications like semantic search and RAG systems.
Endee is a high-performance vector database designed to handle up to 1 billion vectors
on a single node with optimized HNSW indexing.

Retrieval Augmented Generation (RAG) combines information retrieval with language generation.
A user query is first embedded and used to retrieve relevant documents from a vector database.
These documents are then passed as context to a language model to generate accurate answers.
"""


def print_result(result: dict):
    print("\n" + "=" * 60)
    print(f"❓ Question : {result['question']}")
    print(f"📚 Sources  : {', '.join(result['sources'])}")
    print(f"🔢 Chunks   : {result['num_chunks_retrieved']} retrieved")
    print("-" * 60)
    print(f"💬 Answer:\n{result['answer']}")
    print("=" * 60 + "\n")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()
    engine = RAGEngine()

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <file.txt>")
            sys.exit(1)
        filepath = sys.argv[2]
        n = engine.ingest_file(filepath)
        print(f"✅ Ingested {n} chunks from {filepath}")

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python main.py ask \"Your question here\"")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        result = engine.answer(question)
        print_result(result)

    elif command == "demo":
        print("🚀 Running demo — ingesting sample AI knowledge base …\n")
        engine.ingest_text(DEMO_TEXT, source="ai_knowledge_base")

        demo_questions = [
            "What is Retrieval Augmented Generation?",
            "How does Endee work as a vector database?",
            "What is deep learning used for?",
        ]
        for q in demo_questions:
            result = engine.answer(q)
            print_result(result)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
