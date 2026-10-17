import os
from langchain_openrouter import ChatOpenRouter


class LLM:
    """Wraps the LLM model served through OpenRouter."""

    def __init__(self):
        self.llm = ChatOpenRouter(
            model=os.environ["LLM_MODEL"],
            temperature=0,
            max_tokens=1000,
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["OPEN_ROUTER_API_KEY"],
        )
