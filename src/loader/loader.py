"""Turn the files in the knowledge folder into LangChain Documents."""

from pathlib import Path
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPDFLoader,
)
from langchain_core.documents import Document

# PDFs in these sub-folders are scanned images → use OCR
SCAN_FOLDERS = {"scans"}


class Loader:
    """Recursively loads all supported files from a root directory into LangChain Documents."""

    def __init__(self, root: Path):
        self.root = root

    def load(self) -> list[Document]:
        return self.load_corpus()

    @staticmethod
    def _pdf(path: Path):
        if path.parent.name in SCAN_FOLDERS:
            return UnstructuredPDFLoader(str(path), mode="elements", strategy="ocr_only")
        return PyMuPDFLoader(str(path))

    def load_file(self, path: Path) -> list[Document]:
        docs = LOADER_FOR[path.suffix.lower()](path).load()
        for doc in docs:
            doc.metadata.update(
                filename=path.name,
                area=path.parent.name,
                doc_type=path.suffix.lstrip(".").lower(),
            )
        return docs

    def load_corpus(self) -> list[Document]:
        print(f"Loading documents from {self.root} ...")
        docs, failed = [], []
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in LOADER_FOR:
                continue
            try:
                docs.extend(self.load_file(path))
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")
        print(f"Loaded {len(docs)} documents from {self.root}")
        for line in failed:
            print("  skipped", line)
        return docs


LOADER_FOR = {
    ".pdf": Loader._pdf,
    ".md": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
    ".csv": lambda p: CSVLoader(str(p), encoding="utf-8", metadata_columns=["product", "category"]),
}
