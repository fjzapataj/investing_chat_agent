import os
import warnings
from dotenv import load_dotenv
import logging
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_ollama import OllamaEmbeddings
import tiktoken

from app.storage.vector_storage import VectorStorage

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

load_dotenv()

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)

defaul_config = {
    "vdb_config": {
        "index_name": "youtube-transcription-summary-ollama-index",
        "embedding_model": embedding_model,
    },
    "source_folder_path": "data/pending",
}


class DocumentLoader:
    """Class to ingest documents from a csv file and store them in a vector store"""

    def __init__(
        self,
        config: dict = defaul_config,
    ):
        self.source_folder_path = config.get("source_folder_path")
        self.vdb_config = config.get("vdb_config")
        self.vector_store_manager = VectorStorage(self.vdb_config)

    def ingest_documents(self, split_docsuments: bool = False):
        documents, file_paths = self.load_documents_from_csv()

        if split_docsuments:
            documents = self.split_documents(documents)
        logging.info(f"Loaded {len(documents)} documents")
        ids = self.add_documents(docs=documents)
        logging.info(f"Added {len(documents)} documents")
        if len(ids) == len(documents):
            logging.info("All documents added successfully")
            self.move_documents(file_paths)
            logging.info(f"Moved {len(file_paths)} documents to processed folder")
        else:
            logging.warning(
                f"Not all documents were added successfully. {len(ids)} out of {len(documents)}"
            )

    def split_documents(
        self, documents: list, chunk_size: int = 2000, chunk_overlap: int = 200
    ):
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_splitter = TokenTextSplitter(
            encoding_name="o200k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        chunks = text_splitter.split_documents(documents)

        previous_video_id = None
        sequence_index = 1
        for i, chunk in enumerate(chunks):
            current_video_id = chunk.metadata.get("videoId")
            if current_video_id != previous_video_id:
                sequence_index = 1

            chunk.metadata["sequence_index"] = sequence_index
            sequence_index += 1
            previous_video_id = current_video_id
            chunk.metadata["len_tokens_chunk"] = len(
                encoding.encode(chunk.page_content)
            )

        return chunks

    def add_documents(self, docs: list) -> list:
        if len(docs) == 0:
            logging.warning("No documents to add")
            return []
        ids = self.vector_store_manager.add_documents(docs)
        logging.info(f"Added {len(ids)} documents")
        return ids

    def load_documents_from_csv(self):
        all_documents = []
        file_paths = []
        for root, dirs, files in os.walk(self.source_folder_path):
            for file in files:
                if file.endswith(".csv"):
                    path = os.path.join(root, file)
                    logging.info(f"Loading data from {path}")
                    loader = CSVLoader(
                        file_path=path,
                        csv_args={"delimiter": ";"},
                        content_columns=["title", "summary"],
                        source_column="videoUrl",
                        metadata_columns=[
                            "videoId",
                            "title",
                            "publish_date",
                            "assets",
                            "publishTime",
                            "kind",
                            "channelId",
                            "channelTitle",
                            "duration",
                        ],
                    )
                    data = loader.load()
                    all_documents = all_documents + data
                    file_paths.append(path)
        logging.info(f"Loaded {len(all_documents)} documents")
        return all_documents, file_paths

    def move_documents(self, file_paths: list):
        for path in file_paths:
            processed_path = path.replace("pending", "processed")
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            os.rename(path, processed_path)
        logging.info(f"Moved {len(file_paths)} documents to processed folder")
        return file_paths
