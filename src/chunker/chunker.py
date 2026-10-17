"""Turn Documents into chunks using a recursive character text splitter."""

import hashlib
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """Splits documents into overlapping chunks and generates deterministic IDs."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def chunk_documents(self, data):
        return self.splitter.split_documents(data)

    @staticmethod
    def chunk_id(chunk) -> str:
        """Generate a deterministic UUID from the chunk content (deduplication-friendly)."""
        digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))

    def get_chunks_with_ids(self, data, metadata=None):
        chunks = self.chunk_documents(data)
        ids = [self.chunk_id(c) for c in chunks]
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
        print(f"{len(ids)} IDs, {len(set(ids))} unique — duplicates: {len(ids) != len(set(ids))}")
        return chunks, ids
