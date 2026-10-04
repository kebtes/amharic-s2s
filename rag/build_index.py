import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.document_loader import BankDocumentLoader
from rag.text_processor import AmharicTextProcessor
from rag.embeddings import AmharicEmbeddingModel
from rag.retriever import HybridRetriever
from rag.rag_engine import AmharicBankRAG


def main():
    print("=== Amharic Bank Document RAG - Index Builder ===")
    
    faq_dir = "Amharic FAQ"
    text_dir = "amharic text"
    output_dir = "vector_store"

    t_start = time.time()

    # 1. Load documents
    print("\n[Step 1/4] Loading documents from disk...")
    loader = BankDocumentLoader(faq_dir=faq_dir, text_dir=text_dir)
    documents = loader.load_all_documents()
    if not documents:
        print("Error: No documents found! Please check document directory paths.")
        sys.exit(1)

    # 2. Process and Chunk
    print("\n[Step 2/4] Processing and chunking Amharic text...")
    processor = AmharicTextProcessor(chunk_size=400, chunk_overlap=60)
    chunks = processor.process_all(documents)
    if not chunks:
        print("Error: No chunks created from documents!")
        sys.exit(1)

    # 3. Embed & Index
    print("\n[Step 3/4] Generating embeddings and building hybrid index...")
    embedder = AmharicEmbeddingModel(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    retriever = HybridRetriever(embedder=embedder)
    retriever.build_index(chunks)

    # 4. Save Index
    print(f"\n[Step 4/4] Saving vector store to '{output_dir}'...")
    retriever.save(output_dir)

    t_total = time.time() - t_start
    print(f"\n✅ Indexing pipeline complete in {t_total:.2f} seconds!")

    # Verify with sample queries
    print("\n--- Sanity Verification Query ---")
    rag = AmharicBankRAG(index_dir=output_dir, embedder=embedder)
    test_query = "የካርድ አገልግሎት ክፍያ ስንት ነው?"
    res = rag.query(test_query, top_k=2)
    print(f"Test Query: {test_query}")
    print(f"Retrieval Latency: {res['latency_ms']} ms")
    print(f"Sources: {res['sources']}")
    print(f"Formatted Prompt Snippet:\n{res['formatted_prompt'][:300]}...")


if __name__ == "__main__":
    main()
