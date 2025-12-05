import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
log = CustomLogger().get_logger(__name__)

class ModelLoader:
    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config = load_config()
        log.info("Configuration loaded successfully", config_keys=list(self.config.keys()))


    def _validate_env(self):
        required_vars = ["GROQ_API_KEY","OPENAI_API_KEY","GOOGLE_API_KEY"]
        self.api_keys = {key:os.getenv(key) for key in required_vars}
        missing = [k for k, v in self.api_keys.items() if not v]
        if missing:
            log.error("Missing required environment variables", missing_vars=missing)
            raise DocumentPortalException("missing environment variables", sys)
        
        log.info("Environment variables validated", available_keys=[k for k in self.api_keys if self.api_keys[k]])


    def load_embeddings(self):
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return GoogleGenerativeAIEmbeddings(model=model_name,
                                                google_api_key=self.api_keys.get("GOOGLE_API_KEY")) #type: ignore
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "groq")

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_keys.get("GROQ_API_KEY"), #type: ignore
                temperature=temperature,
            )

        # elif provider == "openai":
        #     return ChatOpenAI(
        #         model=model_name,
        #         api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
        #         temperature=temperature,
        #         max_tokens=max_tokens
        #     )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")