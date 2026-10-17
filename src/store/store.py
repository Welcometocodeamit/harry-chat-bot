import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from .embedding import Embedding


class QdrantStore:
    """Manages the Qdrant vector store: connection, collection creation, and document storage."""

    def __init__(self, collection_name: str = "hogwarts"):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.environ["QDRANT_CLIENT_URL"],
            api_key=os.environ.get("QDRANT_API_KEY") or None,
        )
        self.embeddings = Embedding().embeddings

        # Probe the embedding dimension so the collection is created correctly
        self.dimension = len(self.embeddings.embed_query("probe"))
        print(f"Embedding dimension: {self.dimension}")

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
