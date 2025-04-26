from typing import Iterable, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError
import yfinance

from langchain_community.document_loaders.web_base import WebBaseLoader


class YahooFinanceNewsInput(BaseModel):
    """Input for the YahooFinanceNews tool."""

    query: str = Field(description="company ticker query to look up")


class YahooFinanceNewsTool(BaseTool):  # type: ignore[override, override]
    """Tool that searches financial news on Yahoo Finance."""

    name: str = "yahoo_finance_news"
    description: str = (
        "Useful for when you need to find financial news "
        "about a public company. "
        "Input should be a company ticker. "
        "For example, AAPL for Apple, MSFT for Microsoft."
    )
    top_k: int = 10
    """The number of results to return."""

    args_schema: Type[BaseModel] = YahooFinanceNewsInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Yahoo Finance News tool."""

        try:
            company = yfinance.Ticker(query)
            if company.isin is None:
                return f"Company ticker {query} not found."
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"Company ticker {query} not found."

        all_news = []
        try:
            # Iterar sobre las noticias y extraer los atributos de interés
            for news_item in company.news:
                # Crear un diccionario para almacenar los metadatos de cada noticia
                content = news_item["content"]
                link = content.get("canonicalUrl").get("url")

                loader = WebBaseLoader(web_path=link)
                doc = loader.load()

                metadata = {
                    "contentType": content.get("contentType"),
                    "title": content.get("title"),
                    "description": content.get("description"),
                    "summary": content.get("summary"),
                    "pubDate": content.get("pubDate"),
                    "displayTime": content.get("displayTime"),
                    "url": link,
                }
                doc[0].metadata = metadata

                all_news.append(doc[0])
        except (HTTPError, ReadTimeout, ConnectionError):
            if not all_news:
                return f"No news found for company that searched with {query} ticker."
        if not all_news:
            return f"No news found for company that searched with {query} ticker."

        result = self._format_results(all_news, query)
        if not result:
            return f"No news found for company that searched with {query} ticker."
        return result

    @staticmethod
    def _format_results(docs: Iterable[Document], query: str) -> str:
        doc_strings = [
            "\n".join(
                # [doc.metadata["title"], doc.metadata["description"], doc.page_content]
                [doc.metadata["title"], doc.page_content.strip()]
            )
            for doc in docs
            if query in doc.metadata["description"] or query in doc.metadata["title"]
        ]
        return "\n\n".join(doc_strings)


class YahooFinanceNewsSummaryTool(BaseTool):
    """Tool that searches summaries financial news on Yahoo Finance."""

    name: str = "yahoo_finance_news_summary"
    description: str = (
        "Useful for when you need to find financial news summaries"
        "about a public company. "
        "Input should be a company ticker. "
        "For example, AAPL for Apple, MSFT for Microsoft."
    )
    top_k: int = 10
    """The number of results to return."""

    args_schema: Type[BaseModel] = YahooFinanceNewsInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Yahoo Finance News tool."""
        company = yfinance.Ticker(query)
        try:
            if company.isin is None:
                return f"Company ticker {query} not found."
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"Company ticker {query} not found."

        all_news = []

        try:
            # Iterar sobre las noticias y extraer los atributos de interés
            for news_item in company.news:
                content = news_item["content"]
                metadata = {
                    "contentType": content.get("contentType"),
                    "title": content.get("title"),
                    "description": content.get("description"),
                    "summary": content.get("summary"),
                    "pubDate": content.get("pubDate"),
                    "displayTime": content.get("displayTime"),
                    "url": content.get("canonicalUrl").get("url"),
                }

                all_news.append(metadata)

        except Exception:
            return f"No news found for company that searched with {query} ticker."

        if not all_news:
            return f"No news found for company that searched with {query} ticker."

        result = self._format_results(all_news, query)
        if not result:
            return f"No news found for company that searched with {query} ticker."
        return result

    @staticmethod
    def _format_results(docs: list, query: str) -> str:
        doc_strings = [
            "\n".join([doc.get("title"), doc.get("summary")])
            for doc in docs
            if query in doc.get("description") or query in doc.get("title")
        ]
        return "\n\n".join(doc_strings)
