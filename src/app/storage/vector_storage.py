import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


class VectorStorage:
    def __init__(self, config: dict):
        self.index_name = config.get("index_name")
        self.embedding_model = config.get("embedding_model")
        if self.index_name is None:
            raise ValueError("Index name must be provided")
        if self.embedding_model is None:
            raise ValueError("Embedding model must be provided")
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embedding_model,
            pinecone_api_key=PINECONE_API_KEY,
        )

    def add_documents(self, docs: list) -> list:
        return self.vectorstore.add_documents(docs)

    def add_texts(self, texts: list[str], metadata=None) -> list:
        return self.vectorstore.from_texts(
            texts, metadatas=metadata, embedding=self.embedding_model
        )

    def search(self, query: str, k: int = 5, metadata_filter: dict = None) -> list:
        return self.vectorstore.similarity_search(query, k=k, filter=metadata_filter)

    def delete(self, ids) -> None:
        self.vectorstore.delete(ids)
