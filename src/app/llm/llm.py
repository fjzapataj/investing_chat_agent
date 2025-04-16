import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)


from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

default_config = {
    "type": "ollama",
    "model": "phi4:latest",
    "base_url": "http://localhost:11434",
}


class ExtractAssets(BaseModel):
    activos_mencionados: Optional[List[str]] = Field(
        description="Lista de los activos mencionados para invertir", default=None
    )
    tickers_activos: Optional[Dict[str, str]] = Field(
        description="Tickers de activos mensionados, si no tiene ticker poner None",
        default=None,
    )


class Llm:
    def __init__(self, config: dict = default_config):
        self.type = config.get("type")
        if self.type == "ollama":
            self.base_url = config.get("base_url")
            self.model = config.get("model")
            self.llm = ChatOllama(base_url=self.base_url, model=self.model)
        elif self.type == "openai":
            self.api_key = config.get("api_key")
            self.model = config.get("model", "gpt-4o-mini")
            self.temperature = config.get("temperature", 0)
            self.max_tokens = config.get("max_tokens")
            self.model = config.get("model")
            self.llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # base_url="...",
            )
        else:
            raise ValueError(
                "Unsupported LLM type. Supported types are: ollama, openai"
            )

    def get_llm(self):
        return self.llm

    def load_prompt(self, prompt_path: str):
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt no encontrado: {prompt_path}")
        with open(prompt_path) as f:
            prompt_text = f.read()
        return prompt_text


class SummaryLlm(Llm):
    def __init__(self, config: dict = default_config):
        super().__init__(config)
        self.parser = PydanticOutputParser(pydantic_object=ExtractAssets)

        self.summary_prompt_template = self.load_prompt(
            prompt_path="app/prompt/v1_summary_asistance.txt"
        )
        self.extract_info_prompt_template = self.load_prompt(
            prompt_path="app/prompt/v1_extract_info.txt"
        )

    def summarize(self, context):
        template = PromptTemplate(
            template=self.summary_prompt_template,
            input_variables=["context"],
        )

        qna_chain = template | self.llm | StrOutputParser()
        return qna_chain.invoke({"context": context})

    def extract_assets(self, context):
        template = PromptTemplate(
            template=self.extract_info_prompt_template,
            input_variables=["texto"],
            partial_variables={
                "format_instruction": self.parser.get_format_instructions()
            },
        )
        qna_chain = template | self.llm | self.parser
        return qna_chain.invoke({"context": context})


class ChatLlm(Llm):
    def __init__(self, config: dict = default_config):
        super().__init__(config)
        self.prompt_template = Llm.load_prompt(
            prompt_path="app/prompt/v1_expert_asistance.txt"
        )

    def ask_llm(self, context, question):
        template = ChatPromptTemplate(template_format=self.prompt_template)

        qna_chain = template | self.llm | StrOutputParser()
        return qna_chain.invoke({"context": context, "question": question})
