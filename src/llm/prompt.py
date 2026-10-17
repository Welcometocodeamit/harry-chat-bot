from langchain_core.prompts import ChatPromptTemplate


class Prompt:
    """Defines the RAG prompt template for the Harry Potter assistant."""

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a Harry Potter knowledge assistant.
Answer using only the context below.

Rules:
- Answer the question using the provided context.
- The answer may require combining information from multiple passages.
- Do not require a single passage to explicitly state the complete answer.
- Infer causal relationships when directly supported by multiple passages.
- If the context genuinely lacks the necessary information, say that you cannot determine the answer from the context.
- Provide a concise explanation and identify the relevant evidence.

Context:
{context}""",
            ),
            ("human", "{question}"),
        ])
