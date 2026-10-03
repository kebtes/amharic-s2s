import os
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.document_loader import BankDocumentLoader, Document
from rag.text_processor import AmharicTextProcessor
from rag.retriever import HybridRetriever, amharic_tokenizer
from rag.embeddings import AmharicEmbeddingModel


class TestAmharicRAG(unittest.TestCase):
    def setUp(self):
        self.processor = AmharicTextProcessor(chunk_size=200, chunk_overlap=30)

    def test_text_cleaning(self):
        dirty_text = "የባንክ   አገልግሎት  \n\n\n  ጥያቄ ፡ "
        clean = self.processor.clean_text(dirty_text)
        self.assertNotIn("  ", clean)
        self.assertEqual(clean, "የባንክ አገልግሎት\n\nጥያቄ ፡")

    def test_amharic_tokenizer(self):
        text = "የባንክ ሂሳብ እንዴት መክፈት እችላለሁ?"
        tokens = amharic_tokenizer(text)
        self.assertIn("የባንክ", tokens)
        self.assertIn("መክፈት", tokens)

    def test_faq_extraction(self):
        faq_content = """
        ጥያቄ 1: የባንክ ሂሳብ ለመክፈት ምን ያስፈልጋል?
        መልስ: የታደሰ መታወቂያ እና 2 ጉርድ ፎቶግራፍ ያስፈልጋል።

        ጥያቄ 2: የሞባይል ባንኪንግ አገልግሎት እንዴት ይከፈታል?
        መልስ: ቅርንጫፍ በመሄድ ማመልከቻ መሙላት ያስፈልጋል።
        """
        doc = Document(content=faq_content, metadata={"category": "FAQ", "source_file": "faq_test.docx", "file_stem": "faq_test"})
        chunks = self.processor.process_document(doc)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("መታወቂያ", chunks[0].content)

    def test_general_chunking(self):
        text = "የአቢሲኒያ ባንክ የብድር አገልግሎቶች የተለያዩ ናቸው። የንግድ ብድር ለትላልቅ ድርጅቶች ይሰጣል። " * 5
        doc = Document(content=text, metadata={"category": "General", "source_file": "credit.docx", "file_stem": "credit"})
        chunks = self.processor.process_document(doc)
        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
