import os
from langchain_openai import OpenAIEmbeddings


class Embedding:
    """Wraps the OpenAI-compatible embedding model served through OpenRouter."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=os.environ["EMBEDDING_MODEL"],
            openai_api_base=os.environ["OPEN_API_BASE"],
            api_key=os.environ["OPEN_ROUTER_API_KEY"],
        )
