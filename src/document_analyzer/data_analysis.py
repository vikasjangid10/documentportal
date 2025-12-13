import os
import sys

from logger.custom_logger import CustomLogger    
from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from model.models import Metadata

from langchain_core.output_parsers import JsonOutputParser
from prompt.prompt_library import prompt


class DocumentAnalyzer:
    """
    Analyzes documents using a pre-trained model.
    Automatically logs all actions and supports session-based organization
    """

    def __init__(self):
        self.log = CustomLogger().get_logger(__name__)

        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()

            # JSON parser with Pydantic schema
            self.parser = JsonOutputParser(
                pydantic_object=Metadata
            )

            self.prompt = prompt

            self.log.info("DocumentAnalyzer initialized successfully")

        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DocumentPortalException(
                "Error initializing DocumentAnalyzer",
                sys
            )

    def analyze_metadata(self, document_text: str) -> dict:
        try:
            # Build chain (LangChain 1.x style)
            chain = self.prompt | self.llm | self.parser
            self.log.info("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instruction": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            self.log.info(
                "Metadata extraction successful",
                keys=list(response.keys())
            )

            return response

        except Exception as e:
            self.log.error("Error analyzing document metadata", error=str(e))
            raise DocumentPortalException(
                "Metadata extraction failed"
            ) from e
