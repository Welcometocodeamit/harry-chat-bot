import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm.prompt import Prompt
from llm.llm import LLM
from loader.loader import Loader
from chunker.chunker import Chunker
from store.store import QdrantStore
from settings import KNOWLEDGE_DIR


class Chain:
    """Orchestrates the full RAG pipeline: retrieval → prompt → LLM → answer."""

    def __init__(self):
        self.store = QdrantStore()
        self.retriever = self.store.vector_store.as_retriever(
            search_type=os.environ.get("SEARCH_TYPE", "similarity"),
            search_kwargs={"k": int(os.environ.get("SEARCH_K", 15))},
        )
        self.llm = LLM().llm
        self.prompt = Prompt().prompt
        self.rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def load_knowledge_to_db(self, batch_size: int = 500):
        """Load documents from the knowledge directory, chunk them, and upsert into Qdrant."""
        docs = Loader(KNOWLEDGE_DIR).load_corpus()
        chunks, ids = Chunker().get_chunks_with_ids(docs, metadata={"tenant_id": "harry_potter"})
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            self.store.vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
            print(f"  Uploaded batch {i // batch_size + 1} ({len(batch_chunks)} chunks)")
        print(f"Loaded {len(chunks)} chunks into Qdrant collection '{self.store.vector_store.collection_name}'")

    @staticmethod
    def format_docs(docs) -> str:
        """Format retrieved documents into a numbered context string for the prompt."""
        return "\n\n---\n\n".join(
            f"[{i}] {d.metadata.get('filename', 'unknown')}\n{d.page_content}"
            for i, d in enumerate(docs, start=1)
        )

    def answer_question(self, question: str) -> str:
        """Answer a question and return the full response string."""
        response = ""
        for token in self.rag_chain.stream(question):
            response += token
        return response

    def answer_question_stream(self, question: str):
        """Stream answer tokens one by one (for real-time UI rendering)."""
        for token in self.rag_chain.stream(question):
            yield token
