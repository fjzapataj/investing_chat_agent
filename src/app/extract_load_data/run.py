import pandas as pd
import logging
from datetime import datetime
from langchain_ollama import OllamaEmbeddings

from src.app.extract_load_data.youtube_ingest import YoutubeIngest
from src.app.llm.llm import SummaryLlm
from src.app.extract_load_data.loader_docs import DocumentLoader

logging.basicConfig(level=logging.INFO)

default_llm_config = {
    "type": "ollama",
    "model": "phi4:latest",  #'phi4:latest'#'llama3.2:3b'#phi4:latest#gemma3:27b#"qwq:latest"
    "base_url": "http://localhost:11434",
}


default_loader_config = {
    "vdb_config": {
        "index_name": "youtube-transcription-summary-ollama-index",
        "embedding_model": OllamaEmbeddings(
            model="nomic-embed-text", base_url="http://localhost:11434"
        ),
    },
    "source_folder_path": "src/data/pending",
}


class YoutubeLoaderRunner:
    """Class to run the Youtube Ingest"""

    def __init__(
        self,
        llm_config: dict = default_llm_config,
        loader_config: dict = default_loader_config,
    ):
        self.llm_config = llm_config
        self.loader_config = loader_config
        self.youtube_ingest = YoutubeIngest()
        self.summary_llm = SummaryLlm(config=self.llm_config)
        self.document_loader = DocumentLoader(config=self.loader_config)

    def run(self, daysback: int = 0):
        channle_list = self.youtube_ingest.ingest_youtube_videos_from_channels(
            daysback=daysback
        )
        df_videos = pd.DataFrame(channle_list)
        df_videos.dropna(subset=["transcript"], inplace=True)
        df_videos["publish_date"] = pd.to_datetime(df_videos["publishTime"]).dt.date
        df_videos = self.run_summary(df_videos)
        self.save_to_csv(df_videos, "src/data/pending/")
        self.document_loader.ingest_documents()

    def run_summary(self, df: pd.DataFrame):
        df["summary"] = df.apply(
            lambda row: self.summary_llm.summarize(row["title"] + row["transcript"]),
            axis=1,
        )
        logging.info("Summary generated for all videos")
        df["assets"] = df.apply(
            lambda row: self._extract_assets_with_retries(row), axis=1
        )
        logging.info("Assets extracted for all videos")
        return df

    def _extract_assets_with_retries(self, row, retries=3):
        for attempt in range(retries):
            try:
                return self.summary_llm.extract_assets(
                    row["title"] + row["transcript"]
                ).activos_mencionados
            except Exception as e:
                if attempt < retries - 1:
                    logging.warning(
                        f"Attempt {attempt + 1} failed for row {row.name}: {e}"
                    )
                else:
                    logging.error(
                        f"Failed to extract assets for row {row.name} after {retries} attempts: {e}"
                    )
                    return None

    def save_to_csv(self, df: pd.DataFrame, file_path: str = "src/data/pending/"):
        date_format = "%Y-%m-%d"
        current_date = datetime.now().strftime(date_format)
        file_name = f"youtube_videos_{current_date}.csv"
        path = file_path + file_name
        df.to_csv(path, index=False, sep=";")
