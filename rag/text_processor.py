import re
from typing import List, Dict, Any
from .document_loader import Document


class Chunk:
    def __init__(self, chunk_id: str, content: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.content = content.strip()
        self.metadata = metadata

    def __repr__(self):
        return f"<Chunk id='{self.chunk_id}' chars={len(self.content)}>"


class AmharicTextProcessor:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text."""
        if not text:
            return ""
        # Remove multiple newlines and spaces
        text = re.sub(r'\r\n|\r', '\n', text)
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        cleaned = "\n".join(lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text by Amharic and standard punctuation boundaries (።, ?, !, \n)."""
        # Split on Amharic full stop (።), question mark (?), newline or double space
        pattern = r'(?<=[።\?\!\n])\s+'
        raw_sentences = re.split(pattern, text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def extract_faq_pairs(self, text: str) -> List[str]:
        """Extract Question-Answer pairs from FAQ structured text."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        qa_pairs = []
        current_block = []

        qa_keywords = [r'^ጥያቄ', r'^Q[:\.\s]', r'^\d+[\.\)]', r'^\?', r'ጥ[:\.]']

        for line in lines:
            is_new_q = any(re.search(kw, line, re.IGNORECASE) for kw in qa_keywords)
            if is_new_q and current_block:
                block_str = "\n".join(current_block)
                if len(block_str) > 15:
                    qa_pairs.append(block_str)
                current_block = [line]
            else:
                current_block.append(line)

        if current_block:
            block_str = "\n".join(current_block)
            if len(block_str) > 15:
                qa_pairs.append(block_str)

        return qa_pairs

    def process_document(self, doc: Document) -> List[Chunk]:
        """Process a Document into semantically meaningful chunks."""
        cleaned_text = self.clean_text(doc.content)
        category = doc.metadata.get("category", "General")
        file_stem = doc.metadata.get("file_stem", "doc")

        chunks = []

        if category == "FAQ":
            # Extract QA blocks
            qa_blocks = self.extract_faq_pairs(cleaned_text)
            if len(qa_blocks) > 1:
                for idx, qa in enumerate(qa_blocks):
                    chunk_meta = dict(doc.metadata)
                    chunk_meta["chunk_index"] = idx
                    chunk_meta["chunk_type"] = "FAQ_PAIR"
                    chunks.append(
                        Chunk(
                            chunk_id=f"{file_stem}_faq_{idx}",
                            content=qa,
                            metadata=chunk_meta
                        )
                    )
                return chunks

        # Fallback / General Text: Sentence-aware sliding window chunking
        sentences = self.split_into_sentences(cleaned_text)
        current_chunk = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = chunk_idx
                chunk_meta["chunk_type"] = "TEXT_BLOCK"
                chunks.append(
                    Chunk(
                        chunk_id=f"{file_stem}_chunk_{chunk_idx}",
                        content=chunk_str,
                        metadata=chunk_meta
                    )
                )
                chunk_idx += 1

                # Keep overlap sentences
                overlap_len = 0
                new_chunk = []
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        new_chunk.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = new_chunk
                current_len = overlap_len

            current_chunk.append(sentence)
            current_len += sentence_len

        if current_chunk:
            chunk_str = " ".join(current_chunk)
            if len(chunk_str) > 10:
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = chunk_idx
                chunk_meta["chunk_type"] = "TEXT_BLOCK"
                chunks.append(
                    Chunk(
                        chunk_id=f"{file_stem}_chunk_{chunk_idx}",
                        content=chunk_str,
                        metadata=chunk_meta
                    )
                )

        return chunks

    def process_all(self, docs: List[Document]) -> List[Chunk]:
        """Process a list of documents into chunks."""
        all_chunks = []
        for doc in docs:
            doc_chunks = self.process_document(doc)
            all_chunks.extend(doc_chunks)
        print(f"Total created chunks: {len(all_chunks)} from {len(docs)} documents.")
        return all_chunks
