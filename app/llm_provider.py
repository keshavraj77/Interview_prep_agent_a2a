"""
LLM Provider Factory
Supports Google Gemini and local LLM (via LM Studio OpenAI-compatible API)
"""
import os
import logging
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

def get_llm_provider(temperature: float = 0.3) -> BaseChatModel:
    """
    Factory function to get the appropriate LLM based on configuration.
    
    Returns:
        Configured LangChain chat model (Gemini or local LLM)
    """
    model_source = os.getenv('MODEL_SOURCE', 'google').lower()
    
    if model_source == 'local':
        from langchain_openai import ChatOpenAI
        
        base_url = os.getenv('LOCAL_LLM_BASE_URL', 'http://127.0.0.1:1234/v1')
        model_name = os.getenv('LOCAL_LLM_MODEL_NAME', 'local-model')
        
        logger.info(f"Initializing local LLM at {base_url}")
        
        return ChatOpenAI(
            base_url=base_url,
            api_key=os.getenv('LOCAL_LLM_API_KEY', 'not-needed'),
            model=model_name,
            temperature=temperature,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        logger.info("Initializing Google Gemini model")
        
        return ChatGoogleGenerativeAI(
            model='gemini-3-flash-preview',
            temperature=temperature
        )
